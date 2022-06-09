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


SPEX_info SPEX_QR_transGS
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
    
    SPEX_matrix *A_T;
    // Final matrices Q and R
    SPEX_matrix *Q, *R;

    // Allocate A_T
    SPEX_CHECK(SPEX_matrix_allocate(&A_T, SPEX_DENSE, SPEX_MPZ, n, m, n*m,
        false, true, NULL));
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            // A'(i,j) = A(j,i)
            SPEX_CHECK(SPEX_mpz_set( SPEX_2D(A_T, i, j, mpz),
                          SPEX_2D(A,   j, i, mpz)));
        }
    }
    
    // Allocate R. We are performing the Thin REF QR factorization so 
    // R is n*n
    SPEX_CHECK(SPEX_matrix_allocate(&R, SPEX_DENSE, SPEX_MPZ, n, n, n*n,
        false, true, NULL));
    
    // Set Q = A
     SPEX_CHECK(SPEX_matrix_copy(&Q, SPEX_DENSE, SPEX_MPZ, A_T, NULL));


     // Compute 1st row of R
    for (j = 0; j < n; j++)
    {
      // R(0,j) = Q(0,:) dot A(:,j)
        for (i = 0; i < n; i++)
        {
            SPEX_CHECK(SPEX_mpz_addmul(SPEX_2D(R,0,j, mpz), SPEX_2D(Q, 0, i, mpz),
                        SPEX_2D(A, i, j, mpz)));
        }
    }
  

     // Perform Factorization
    for (k = 1; k < m; k++)
    {
        // Compute kth column of Q
        for (i = 0; i < n; i++)
        {
            // Q(k,i) = R(j,j)*Q(k,i)
                SPEX_CHECK(SPEX_mpz_mul( SPEX_2D(Q, k, i, mpz),
                              SPEX_2D(Q, k, i, mpz),
                              SPEX_2D(R, 0, 0, mpz)));
                // Q(k,i) = Q(k,i) - R(j,k)*Q(j,i)
                SPEX_CHECK(SPEX_mpz_submul( SPEX_2D(Q, k, i, mpz),
                                 SPEX_2D(R, 0, k, mpz),
                                 SPEX_2D(Q, 0, i, mpz)));
            for (j = 1; j < k; j++)
            {
                // Q(k,i) = R(j,j)*Q(k,i)
                SPEX_CHECK(SPEX_mpz_mul( SPEX_2D(Q, k, i, mpz),
                              SPEX_2D(Q, k, i, mpz),
                              SPEX_2D(R, j, j, mpz)));
                // Q(k,i) = Q(k,i) - R(j,k)*Q(j,i)
                SPEX_CHECK(SPEX_mpz_submul( SPEX_2D(Q, k, i, mpz),
                                 SPEX_2D(R, j, k, mpz),
                                 SPEX_2D(Q, j, i, mpz)));
                SPEX_CHECK(SPEX_mpz_divexact( SPEX_2D(Q, k, i, mpz),
                                       SPEX_2D(Q, k, i, mpz),
                                       SPEX_2D(R, j-1, j-1, mpz)));
            }

        }

        //Compute kth row of R
        for (j = k; j < n; j++)
        {
          // R(k,j) = Q(:,k) dot A(:,j)
          // R(k,j) = Q_T(k,:)*A(:,j)
            for (i = 0; i < n; i++)
            {
                SPEX_CHECK(SPEX_mpz_addmul(SPEX_2D(R,k,j, mpz), SPEX_2D(Q, k, i, mpz),
                            SPEX_2D(A, i, j, mpz)));
            }
        }
    }
    

    (*Q_handle) = Q;
    (*R_handle) = R;
    return SPEX_OK;
}
