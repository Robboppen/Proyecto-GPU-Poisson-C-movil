/*
 * Proyecto: ecuación de Poisson comóvil en 2D (versión CPU de referencia)
 * Modificación de poisson_cpu.c para el potencial gravitatorio peculiar.
 *
 * Ecuación física (coordenadas comóviles x, potencial peculiar phi):
 *     lap_x phi = (3/2) Omega_m H0^2 delta / a
 *
 * Unidades de código: x~ = x / L  (caja periódica [0,1) x [0,1)),
 *                     phi~ = phi / (H0 L)^2
 *     =>  lap phi~ = (3/2) (Omega_m / a) delta        (sin constantes)
 *
 * Cambios respecto al código del profesor:
 *   1. Bordes periódicos: N puntos por lado, h = 1/N, índices con módulo.
 *      Se actualizan TODOS los puntos (no hay borde fijo).
 *   2. Fuente f = (3/2)(Om/a)(delta - <delta>): se resta el promedio para
 *      cumplir la condición de compatibilidad  sum(f) = 0.
 *   3. El potencial queda definido salvo una constante: se fija <phi> = 0.
 *   4. SOR con w_opt periódico: el modo de error más lento es (1,0),
 *      k = (2 pi, 0), con factor de Jacobi rho_J = [1 + cos(2 pi h)]/2
 *      = cos^2(pi h). Entonces w = 2 / (1 + sqrt(1 - rho_J^2))
 *      = 2 / (1 + sin(pi h) sqrt(1 + cos^2(pi h))).
 *   5. Dos fuentes posibles:
 *        modo 0: test con un solo modo, delta = A cos(2 pi m x) cos(2 pi n y)
 *                (solución discreta y continua exactas conocidas)
 *        modo 1: delta leído de 'delta.bin' (p.ej. campo gaussiano
 *                generado con generar_delta.py)
 *
 * Compilar:  gcc -O2 poisson_comovil_cpu.c -lm -o poisson_comovil_cpu.x
 * Ejecutar:  ./poisson_comovil_cpu.x [N] [metodo] [tol] [modo] [Om] [a]
 *            N debe ser par (rojo-negro periódico).
 * Salida:    salida_cpu.bin con dos campos: phi y delta (formato del profesor)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

typedef double real;

double segundos(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void escribir_salida(const char *nombre, int nx, int ny, int ncampos,
                     real **campos) {
  FILE *f = fopen(nombre, "wb");
  int cabecera[4] = {nx, ny, ncampos, (int)sizeof(real)};
  fwrite(cabecera, sizeof(int), 4, f);
  for (int c = 0; c < ncampos; c++)
    fwrite(campos[c], sizeof(real), (size_t)nx * ny, f);
  fclose(f);
}

// Lee delta desde un archivo con el mismo formato (cabecera + 1 campo)
int leer_delta(const char *nombre, real *delta, int N) {
  FILE *f = fopen(nombre, "rb");
  if (!f) return 1;
  int cab[4];
  if (fread(cab, sizeof(int), 4, f) != 4 || cab[0] != N || cab[1] != N ||
      cab[3] != (int)sizeof(real)) {
    fclose(f);
    return 2;
  }
  size_t n = fread(delta, sizeof(real), (size_t)N * N, f);
  fclose(f);
  return (n == (size_t)N * N) ? 0 : 3;
}

// Vecinos periódicos
static inline int mas(int i, int N) { return (i + 1 == N) ? 0 : i + 1; }
static inline int menos(int i, int N) { return (i == 0) ? N - 1 : i - 1; }

// Suma de los 4 vecinos del punto (i,j) con bordes periódicos
static inline real vecinos(const real *u, int i, int j, int N) {
  return u[j * N + mas(i, N)] + u[j * N + menos(i, N)] +
         u[mas(j, N) * N + i] + u[menos(j, N) * N + i];
}

void iteracion_jacobi(const real *u, real *u_new, const real *f, int N,
                      real h2) {
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      u_new[j * N + i] = 0.25 * (vecinos(u, i, j, N) - h2 * f[j * N + i]);
}

// Media iteración rojo-negro sobre los puntos con (i + j) % 2 == color
void media_iteracion_rb(real *u, const real *f, int N, real h2, real w,
                        int color) {
  for (int j = 0; j < N; j++)
    for (int i = (j + color) % 2; i < N; i += 2) {
      int idx = j * N + i;
      real gs = 0.25 * (vecinos(u, i, j, N) - h2 * f[idx]);
      u[idx] += w * (gs - u[idx]);
    }
}

real norma_residuo(const real *u, const real *f, int N, real h2) {
  real suma = 0.0;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      real lap = (vecinos(u, i, j, N) - 4.0 * u[j * N + i]) / h2;
      real r = f[j * N + i] - lap;
      suma += r * r;
    }
  return sqrt(suma);
}

real norma(const real *f, int N) {
  real suma = 0.0;
  for (int k = 0; k < N * N; k++) suma += f[k] * f[k];
  return sqrt(suma);
}

real promedio(const real *f, int N) {
  real suma = 0.0;
  for (int k = 0; k < N * N; k++) suma += f[k];
  return suma / ((real)N * N);
}

// Fija el modo cero: <u> = 0
void restar_promedio(real *u, int N) {
  real m = promedio(u, N);
  for (int k = 0; k < N * N; k++) u[k] -= m;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 256;
  int metodo = (argc > 2) ? atoi(argv[2]) : 2;
  real tol = (argc > 3) ? atof(argv[3]) : 1e-8;
  int modo = (argc > 4) ? atoi(argv[4]) : 0;
  real Om = (argc > 5) ? atof(argv[5]) : 0.3;
  real a = (argc > 6) ? atof(argv[6]) : 1.0;

  const char *nombres[3] = {"Jacobi", "Gauss-Seidel rojo-negro",
                            "SOR rojo-negro"};
  if (metodo < 0 || metodo > 2 || N % 2 != 0) {
    fprintf(stderr, "metodo debe ser 0, 1 o 2 y N debe ser par\n");
    return 1;
  }

  real h = 1.0 / N;  // periódico: x_i = i h, i = 0..N-1
  real h2 = h * h;
  // rho_J = cos^2(pi h) (modo (1,0)) => w = 2/(1+sqrt(1-rho_J^2))
  real w = (metodo == 2) ? 2.0 / (1.0 + sin(PI * h) * sqrt(1.0 + pow(cos(PI * h), 2))) : 1.0;
  int chequeo = 100;
  int max_iter = 10000000;

  real *u = (real *)calloc((size_t)N * N, sizeof(real));
  real *u_new = (real *)calloc((size_t)N * N, sizeof(real));
  real *f = (real *)malloc((size_t)N * N * sizeof(real));
  real *delta = (real *)malloc((size_t)N * N * sizeof(real));

  // Parámetros del test de un modo
  const int m = 1, n = 2;
  const real A = 1.0;

  if (modo == 0) {
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        delta[j * N + i] = A * cos(2 * PI * m * i * h) * cos(2 * PI * n * j * h);
  } else {
    int e = leer_delta("delta.bin", delta, N);
    if (e) {
      fprintf(stderr, "error %d leyendo delta.bin (¿mismo N?)\n", e);
      return 1;
    }
  }

  // Fuente comóvil en unidades de código, con promedio restado
  real dmed = promedio(delta, N);
  real pref = 1.5 * Om / a;
  for (int k = 0; k < N * N; k++) f[k] = pref * (delta[k] - dmed);
  real norma_f = norma(f, N);

  printf("Poisson comóvil 2D periódico: N = %d, método = %s", N,
         nombres[metodo]);
  if (metodo == 2) printf(" (w = %.6f)", w);
  printf(", tol = %.1e, Om = %.3f, a = %.3f, <delta> = %.2e\n", tol, Om, a,
         dmed);

  double t0 = segundos();
  int iter = 0;
  real res = 1.0;
  while (iter < max_iter) {
    for (int k = 0; k < chequeo; k++) {
      if (metodo == 0) {
        iteracion_jacobi(u, u_new, f, N, h2);
        real *tmp = u;
        u = u_new;
        u_new = tmp;
      } else {
        media_iteracion_rb(u, f, N, h2, w, 0);
        media_iteracion_rb(u, f, N, h2, w, 1);
      }
    }
    iter += chequeo;
    restar_promedio(u, N);  // evita que el modo cero derive por redondeo
    res = norma_residuo(u, f, N, h2) / norma_f;
    if (res < tol) break;
  }
  double t_total = segundos() - t0;

  printf("Iteraciones: %d, residuo relativo = %.3e\n", iter, res);
  printf("Tiempo de cómputo: %.3f s (%.3f ms por iteración)\n", t_total,
         1e3 * t_total / iter);

  if (modo == 0) {
    // Autovalores del laplaciano discreto y continuo para cos(2pi m x)cos(2pi n y)
    real lam_d = -4.0 / h2 * (pow(sin(PI * m * h), 2) + pow(sin(PI * n * h), 2));
    real lam_c = -4.0 * PI * PI * (m * m + n * n);
    real err_d = 0.0, err_c = 0.0, umax = 0.0;
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        real modo_ij = cos(2 * PI * m * i * h) * cos(2 * PI * n * j * h);
        real ud = pref * A * modo_ij / lam_d;
        real uc = pref * A * modo_ij / lam_c;
        real v = u[j * N + i];
        if (fabs(v - ud) > err_d) err_d = fabs(v - ud);
        if (fabs(v - uc) > err_c) err_c = fabs(v - uc);
        if (fabs(uc) > umax) umax = fabs(uc);
      }
    printf("Error vs. solución discreta exacta (solver):        %.3e\n",
           err_d / umax);
    printf("Error vs. solución continua (discretización, ~h^2): %.3e (h^2 = %.3e)\n",
           err_c / umax, h2);
  }

  real *campos[2] = {u, delta};
  escribir_salida("salida_cpu.bin", N, N, 2, campos);

  free(u);
  free(u_new);
  free(f);
  free(delta);
  return 0;
}
