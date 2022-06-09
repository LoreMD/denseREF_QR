//------------------------------------------------------------------------------
// SPEX_QR/Source/SPEX_QR_GS.c: Integer preserving Gram Schmidt
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021, Chris Lourenco, US Naval Academy, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


/* This code performs REF QR via the integer preserving Gram Schmidt algorithm
 * from paper 2.
 */

# include "spex_qr_internal.h"


SPEX_info SPEX_QR_GS
(
    SPEX_matrix *A,            // Matrix to be factored
    SPEX_matrix **R_handle,    // Null on input, contains R on output
    SPEX_matrix **Q_handle     // Null on input, contains Q on output
)
{
    SPEX_info info;
    int64_t m = A->m, n = A->n;
    ASSERT( m >= n); // A should be transposed if not true
    if (m < n)
        return SPEX_PANIC;
    ASSERT( A != NULL);
    // Only dense for now
    ASSERT( A->type == SPEX_MPZ);
    ASSERT( A->kind == SPEX_DENSE);
    
    // Indices
    int64_t i, j, k;
    
    // Final matrices Q and R
    SPEX_matrix *Q, *R;
    
    // Allocate R. We are performing the Thin REF QR factorization so 
    // R is n*n
    SPEX_CHECK(SPEX_matrix_allocate(&R, SPEX_DENSE, SPEX_MPZ, n, n, n*n,
        false, true, NULL));
    
    // Set Q = A
     SPEX_CHECK(SPEX_matrix_copy(&Q, SPEX_DENSE, SPEX_MPZ, A, NULL));
     

    // Perform Factorization

     // Compute 1st row of R
    for (j = 0; j < n; j++)
    {
      // R(0,j) = Q(:,0) dot A(:,j)
      SPEX_CHECK(SPEX_dense_mat_dot(Q, 0, A, j, SPEX_2D(R,k,j,mpz)));
    }

     // Perform Factorization
    for (k = 1; k < n; k++)
    {
        // Compute kth column of Q
        for (j = 0; j < k; j++)
        {
            for (i = 0; i < m; i++)
            {
                // Q(i,k) = R(j,j)*Q(i,k)
                SPEX_CHECK(SPEX_mpz_mul( SPEX_2D(Q, i, k, mpz),
                              SPEX_2D(Q, i, k, mpz),
                              SPEX_2D(R, j, j, mpz)));
                // Q(i,k) = Q(i,k) - R(j,k)*Q(i,j)
                SPEX_CHECK(SPEX_mpz_submul( SPEX_2D(Q, i, k, mpz),
                                 SPEX_2D(R, j, k, mpz),
                                 SPEX_2D(Q, i, j, mpz)));
                if (j > 0)
                {
                    // Q(i,k) = Q(i,k)/R(j-1,j-1)
                    SPEX_CHECK(SPEX_mpz_divexact( SPEX_2D(Q, i, k, mpz),
                                       SPEX_2D(Q, i, k, mpz),
                                       SPEX_2D(R, j-1, j-1, mpz)));
                }
            }
        }

        //Compute kth row of R
        for (j = k; j < n; j++)
        {
          // R(k,j) = Q(:,k) dot A(:,j)
          SPEX_CHECK(SPEX_dense_mat_dot(Q, k, A, j, SPEX_2D(R,k,j,mpz)));
        }
    }
    
    (*Q_handle) = Q;
    (*R_handle) = R;
    return SPEX_OK;
}
