/*  Copyright (C) 2012 Matt Hagy <hagy@gatech.edu>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
