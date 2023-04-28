using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Bluebit.MatrixLibrary;

namespace DemoQuants
{
    public class Portofolio
    {

        public Portofolio() { }

        public double mv_calculate_mean(Matrix e, Matrix w)
        {
            Matrix tmp = new Matrix(); tmp = e.Transpose() * w;

            return tmp[0, 0];
        }

        public double mv_calculate_variance(Matrix V, Matrix w)
        {
            Matrix tmp = new Matrix(); tmp  = w.Transpose() * V * w;

            return tmp[0, 0];
        }

        public double mv_calculate_st_dev(Matrix V, Matrix w)
        {
            double var = mv_calculate_variance(V, w);
            return Math.Sqrt(var);
        }


        public Matrix mv_calculate_portofolio_given_mean_noconstraint(Matrix e, Matrix V, Matrix r)
        {
            int number_assets = e.Rows;
            Matrix ones = new Matrix(number_assets, 1);
            for (int i = 0; i < number_assets; i++) { ones[i, 0] = 1; }
            Matrix Vinv = V.Inverse();

            Matrix A = new Matrix();  A = (ones.Transpose() * Vinv * e); double a = A[0, 0];
            Matrix B = new Matrix(); B = e.Transpose() * Vinv * e; double b = B[0, 0];
            Matrix C = new Matrix(); C = ones.Transpose() * Vinv * ones; double c = C[0, 0];
            Matrix D = new Matrix(); D =B * C - A * A; double d = D[0, 0];

            Matrix Vinv1 = new Matrix(); Vinv1 = Vinv* ones;
            Matrix Vinve = new Matrix(); Vinve = Vinv* e;
            Matrix g = new Matrix(); g = (Vinv1 * b - Vinve * a) * (1 / d);
            Matrix h = new Matrix(); h = (Vinve * c - Vinv1 * a) * (1 / d);
            Matrix w = new Matrix(); w = g = h * r;

            return w;
        }

    }
}
