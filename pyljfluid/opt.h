#ifndef OPT_H_
#define OPT_H_

#ifdef __GNUC__
#  define GCC_ATTRIBUTE(ATTR) __attribute__(ATTR)
#else
#  define GCC_ATTRIBUTE(ATTR) 
#endif

#ifndef NO_GCC_OPTIMIZATIONS
/* load effective location before use */
#  define prefetch(p) __builtin_prefetch(p, 1, 3)
/* branch prediction hints */
#  define likely(x)   __builtin_expect(!!(x), 1)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#else
#  define prefetch(p) do {} while(0)
#  define likely(x)   (x)
#  define unlikely(x) (x)
#endif

/* compiler optimization requiring unaliased arrays
 * needed for Fortran-like array processing */
#ifndef NRESTRICT
# define OPT_RESTRICT restrict
#else
# define OPT_RESTRICT
#endif

#endif /*OPT_H_*/
