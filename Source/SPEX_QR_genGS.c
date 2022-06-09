//------------------------------------------------------------------------------
// SPEX_QR/Source/SPEX_QR_genGS.c: Integer preserving gen Gram Schmidt by erlingsson
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021, Chris Lourenco, US Naval Academy, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


/* This code performs REF QR via the "exact division" Gram Schmidt algorithm
 * from Erlingsson and Kaltofen.
 */

# include "spex_qr_internal.h"


SPEX_info SPEX_QR_genGS
(
    SPEX_matrix *A,            // Matrix to be factored
    SPEX_matrix **R_handle,    // Null on input, contains R' on output
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
    int64_t i, j, l;
    
    // Final matrices Q and R
    SPEX_matrix *Q, *R, *d; //In paper these are Bt and Mt
    mpz_t s;
    mpz_init(s);
    
    // Allocate R. 
    // R is n*n
    SPEX_CHECK(SPEX_matrix_allocate(&R, SPEX_DENSE, SPEX_MPZ, n, n, n*n,
        false, true, NULL));
    
    // Allocate Q. Algorithm is meant for basis so Q is n*n
    SPEX_CHECK(SPEX_matrix_allocate(&Q, SPEX_DENSE, SPEX_MPZ, n, n, n*n,
        false, true, NULL));
    //Q=zeros
 
    // Allocate d.
    SPEX_matrix_allocate(&d, SPEX_DENSE, SPEX_MPZ, n+1, 1, n+1, false, true, NULL);
    //d(0)=1
    SPEX_CHECK( SPEX_mpz_set_ui(SPEX_1D(d,0,mpz),1));


    // Perform Factorization
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            SPEX_CHECK( SPEX_mpz_set_ui(s,0));
            for (l = 0; l < j; l++)
            {
                //Accumulating the sum to substract from A(i,j)
                //s=(d(l+1)*s+R(i,l)*R(j,l))
                SPEX_CHECK(SPEX_mpz_mul( s,
                              SPEX_1D(d, l+1, mpz),
                              s));
                SPEX_CHECK(SPEX_mpz_addmul( s,
                                 SPEX_2D(R, i, l, mpz),
                                 SPEX_2D(R, j, l, mpz)));
                //s=s/d(l)
                SPEX_CHECK(SPEX_mpz_divexact( s,
                                       s,
                                       SPEX_1D(d, l, mpz)));
            }
            //ATA(i,j)-sum
            //R(i,j)=d(j)*dot(A(:,i),A(:,j))-s
            SPEX_CHECK( SPEX_dense_mat_dot(A, i, A, j, SPEX_2D(R,i,j,mpz)));
            SPEX_CHECK( SPEX_mpz_mul(SPEX_2D(R,i,j, mpz), SPEX_1D(d,j,mpz), SPEX_2D(R,i,j, mpz)));
            SPEX_CHECK( SPEX_mpz_sub(SPEX_2D(R,i,j, mpz), SPEX_2D(R,i,j, mpz), s));
        }  
        

        SPEX_CHECK( SPEX_mpz_set_ui(s,0));
        for (l = 0; l <= i-1; l++)
        {
            //Accumulating the sum for pivot
            //s=d(l+1)*s+R(i,l)*R(j,l))
            SPEX_CHECK(SPEX_mpz_mul( s,
                              SPEX_1D(d, l+1, mpz),
                              s));
            SPEX_CHECK(SPEX_mpz_addmul( s,
                                 SPEX_2D(R, i, l, mpz),
                                 SPEX_2D(R, j, l, mpz)));
            //s=s/d(l)
            SPEX_CHECK(SPEX_mpz_divexact( s,
                                       s,
                                       SPEX_1D(d, l, mpz)));
        }
        //R(i,i)=d(i)*dot(A(:,i),A(:,i))-s
        SPEX_CHECK( SPEX_dense_mat_dot(A, i, A, i, SPEX_2D(R,i,i,mpz)));
        SPEX_CHECK( SPEX_mpz_mul(SPEX_2D(R,i,i, mpz), SPEX_1D(d,i,mpz), SPEX_2D(R,i,i, mpz)));
        SPEX_CHECK( SPEX_mpz_sub(SPEX_2D(R,i,i, mpz), SPEX_2D(R,i,i, mpz), s));

        //d(i+1)=R(i,i)
        SPEX_CHECK( SPEX_mpz_set(SPEX_1D(d,i+1,mpz),SPEX_2D(R,i,i,mpz)));

        // Compute kth column of Q (Pursell)
        for (j = 0; j < n; j++)
        {
            // Q(j,i) = d(1)*A(j,i)
            SPEX_CHECK(SPEX_mpz_mul( SPEX_2D(Q, j, i, mpz),
                              SPEX_2D(A, j, i, mpz),
                              SPEX_1D(d, 1, mpz)));
            // Q(j,i) = Q(j,i) - R(i,0)*A(j,0)
            SPEX_CHECK(SPEX_mpz_submul( SPEX_2D(Q, j, i, mpz),
                                 SPEX_2D(R, i, 0, mpz),
                                 SPEX_2D(A, j, 0, mpz)));
        }
        for (l = 0; l <= i-2; l++)
        {
            for (j = 0; j < n; j++)
            {
                // Q(j,i) = d(l+2)*Q(j,i)
                SPEX_CHECK(SPEX_mpz_mul( SPEX_2D(Q, j, i, mpz),
                              SPEX_2D(Q, j, i, mpz),
                              SPEX_1D(d, l+2, mpz)));
                // Q(j,i) = Q(j,i) - R(i,l+1)*Q(j,l+1)
                SPEX_CHECK(SPEX_mpz_submul( SPEX_2D(Q, j, i, mpz),
                                 SPEX_2D(R, i, l+1, mpz),
                                 SPEX_2D(Q, j, l+1, mpz)));
                // Q(j,i) = Q(j,i)/d(l+1)
                SPEX_CHECK(SPEX_mpz_divexact( SPEX_2D(Q, j, i, mpz),
                                       SPEX_2D(Q, j, i, mpz),
                                       SPEX_1D(d, l+1, mpz)));
                
            }
        }
    }

    //set first column of Q as the first column of A
    for (i = 0; i < n; i++)
    {
        SPEX_CHECK( SPEX_mpz_set(SPEX_2D(Q,i,0, mpz),SPEX_2D(A,i,0,mpz)));   
    }
    
    (*Q_handle) = Q;
    (*R_handle) = R;
    return SPEX_OK;
}
