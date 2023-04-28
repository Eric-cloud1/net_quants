using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace DemoQuants
{
    public class Normal
    {
        public Normal() { }

        private double random_0_1()
        {
            double r = 0;
            Random rnd = new Random(0);
            r = rnd.NextDouble();
            return r;
        }
        //uniformly distributed random variates used for polar method for normal density function
        public double random_normal()
        {
            double u1 =0;
            double u2 =0;
            double V1 = 0;
            double V2 = 0;
            double S = 2;
            while (S >= 1)
            {
                u1 = random_0_1();
                u2 = random_0_1();
                V1 = (2 * u1) - 1;
                V2 = (2 * u2) - 1;
                S = Math.Pow(V1, 2) + Math.Pow(V2, 2);
            }
            double X1 = V1 * Math.Sqrt((-2 * Math.Log(S)) / S);
            return X1;
        }

        // Z = X - M / sigma
        public double random_quote(double sigma, double mu)
        {
            
            return mu + sigma * GetNormal();
        }

        // exp (mu * Delta_t + sigma Sqrt(Delta_t)  Delta_t = 1, mu = 15%, sigma = 30%
        public double logNormal_quote(double lastQuote, double sigma, double mu, double delta_t)
        {
            return lastQuote * Math.Exp(mu * delta_t + sigma * GetNormal() * Math.Sqrt(delta_t));
        }

        // simulated quote based on last quote, interest and volatility
        public double simulate_log_normal_random_variable(double S, double r, double sigma, double t)
        {
            double R = (r - 0.5 * Math.Pow(sigma, 2)) * t;
            double SD = sigma * Math.Sqrt(t);
            return S * Math.Exp(R + SD * random_normal());
        }

        // normal distributed function
        public double n(double z)
        {
            return (1 / Math.Sqrt(2 * Math.PI)) * Math.Exp(-0.5 * z * z);
        }

        // cumulative normal distribution
        public double N(double z)
        {
            if (z > 6.0) return 1;
            if (z < -6) return 0;
            double b1 = 0.31938153;
            double b2 = -0.356563782;
            double b3 = 1.781477937;
            double b4 = -1.821255978;
            double b5 = 1.330274429;
            double p = 0.2316419;
            double c2 = 0.3989423;

            double a = Math.Abs(z);
            double t = 1 / (1 + a * p);
            double b = c2 * Math.Exp((-z) * (z / 2));
            double n = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;
            n = 1 - b * n;
            if (z < 0) n = 1 - n;

            return n;
        }
        private double f(double x, double y, double aprime, double bprime, double rho)
        {
            double r = aprime * (2 * x - aprime) + 
                bprime * (2 * y - bprime) + 2 * rho * (x - aprime) * (y - bprime);
            return Math.Exp(r);
        }
        private double sng(double x)
        {
            if (x > 0) return 1;
            return -1;
        }

        // bivariate normal distribution
        public double N(double a, double b, double rho)
        {
            if(( a <= 0)&&(b<=0)&&(rho<=0)){
                double aprime = a / Math.Sqrt(2*(1 - rho * rho));
                double bprime = b / Math.Sqrt(2 * (1 - rho * rho));
                double[] A = {0.3253030, 0.4211071, 0.1334425,0.006374323};
                double[] B = {0.1337764, 0.6243247, 1.3425378, 2.2626645};
                double sum =0;
                for (int i =0; i<4; i++){
                    for (int j = 0; j < 4; j++)
                    { sum += A[i] * A[j] * f(B[i], B[j], aprime, bprime, rho); }
                } sum = sum * (Math.Sqrt(1 - rho * rho)/Math.PI);
                return sum;
            }
            else if (a * b * rho <= 0)
            {
                if ((a < 0) && (b < 0) && (rho > 0)) {return n(a) - N(a, -b, -rho);}
                else if (( a >=0) &&( b <=0) && ( rho >=0)){ return n(b) - N(-a, b, -rho); }
                else if (( a >=0) &&( b >=0) && ( rho <=0)) 
                {return n(a) + n(b) - 1 + N(-a, -b, rho);}
            }
            else if(a * b * rho >=0){
                double denum = Math.Sqrt(a * a - 2 * rho * a * b + b * b);
                double rho1 = ((rho * a - b) * sng(a))/ denum;
                double rho2 = ((rho * b - a) * sng(b)) / denum;
                double delta = (1 - sng(a) * sng(b)) / 4;
                return N(a, 0, rho1) + N(b, 0, rho2) - delta;
            }
            return -9;

        }

        //RNG
        private static uint m_w = 521288629;
        private static uint m_z = 362436069;

        // The random generator seed can be set three ways:
        // 1) specifying two non-zero unsigned integers
        // 2) specifying one non-zero unsigned integer and taking a default value for the second
        // 3) setting the seed from the system time

        public static void SetSeed(uint u, uint v)
        {
            if (u != 0) m_w = u;
            if (v != 0) m_z = v;
        }

        public static void SetSeed(uint u)
        {
            m_w = u;
        }

        public static void SetSeedFromSystemTime()
        {
            System.DateTime dt = System.DateTime.Now;
            long x = dt.ToFileTime();
            SetSeed((uint)(x >> 16), (uint)(x % 4294967296));
        }

        // This is the heart of the generator.
        // It uses George Marsaglia's MWC algorithm to produce an unsigned integer.
        // See http://www.bobwheeler.com/statistics/Password/MarsagliaPost.txt
        private static uint GetUint()
        {
            m_z = 36969 * (m_z & 65535) + (m_z >> 16);
            m_w = 18000 * (m_w & 65535) + (m_w >> 16);
            return (m_z << 16) + (m_w & 65535);
        }

        public static double GetUniform()
        {
            // 0 <= u <= 2^32
            uint u = GetUint();
            // The magic number below is 1/(2^32 + 2).
            // The result is strictly between 0 and 1.
            return (u + 1) * 2.328306435454494e-10;
        }

    
        // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
        public static double GetNormal()
        {
            // Use Box-Muller algorithm
            double u1 = GetUniform();
            double u2 = GetUniform();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            return r * Math.Sin(theta);
        }

        // Get normal (Gaussian) random sample with specified mean and standard deviation
        public static double GetNormal(double mean, double standardDeviation)
        {
            if (standardDeviation <= 0.0)
            {
                string msg = string.Format("Shape must be positive. Received {0}.", standardDeviation);
                throw new ArgumentOutOfRangeException(msg);
            }
            return mean + standardDeviation * GetNormal();
        }

        // Get exponential random sample with mean 1
        public static double GetExponential()
        {
            return -Math.Log(GetUniform());
        }

        // Get exponential random sample with specified mean
        public static double GetExponential(double mean)
        {
            if (mean <= 0.0)
            {
                string msg = string.Format("Mean must be positive. Received {0}.", mean);
                throw new ArgumentOutOfRangeException(msg);
            }
            return mean * GetExponential();
        }

        public static double GetGamma(double shape, double scale)
        {
            // Implementation based on "A Simple Method for Generating Gamma Variables"
            // by George Marsaglia and Wai Wan Tsang.  ACM Transactions on Mathematical Software
            // Vol 26, No 3, September 2000, pages 363-372.

            double d, c, x, xsquared, v, u;

            if (shape >= 1.0)
            {
                d = shape - 1.0 / 3.0;
                c = 1.0 / Math.Sqrt(9.0 * d);
                for (; ; )
                {
                    do
                    {
                        x = GetNormal();
                        v = 1.0 + c * x;
                    }
                    while (v <= 0.0);
                    v = v * v * v;
                    u = GetUniform();
                    xsquared = x * x;
                    if (u < 1.0 - .0331 * xsquared * xsquared || Math.Log(u) < 0.5 * xsquared + d * (1.0 - v + Math.Log(v)))
                        return scale * d * v;
                }
            }
            else if (shape <= 0.0)
            {
                string msg = string.Format("Shape must be positive. Received {0}.", shape);
                throw new ArgumentOutOfRangeException(msg);
            }
            else
            {
                double g = GetGamma(shape + 1.0, 1.0);
                double w = GetUniform();
                return scale * g * Math.Pow(w, 1.0 / shape);
            }
        }

        public static double GetChiSquare(double degreesOfFreedom)
        {
            // A chi squared distribution with n degrees of freedom
            // is a gamma distribution with shape n/2 and scale 2.
            return GetGamma(0.5 * degreesOfFreedom, 2.0);
        }

        public static double GetInverseGamma(double shape, double scale)
        {
            // If X is gamma(shape, scale) then
            // 1/Y is inverse gamma(shape, 1/scale)
            return 1.0 / GetGamma(shape, 1.0 / scale);
        }

        public static double GetWeibull(double shape, double scale)
        {
            if (shape <= 0.0 || scale <= 0.0)
            {
                string msg = string.Format("Shape and scale parameters must be positive. Recieved shape {0} and scale{1}.", shape, scale);
                throw new ArgumentOutOfRangeException(msg);
            }
            return scale * Math.Pow(-Math.Log(GetUniform()), 1.0 / shape);
        }

        public static double GetCauchy(double median, double scale)
        {
            if (scale <= 0)
            {
                string msg = string.Format("Scale must be positive. Received {0}.", scale);
                throw new ArgumentException(msg);
            }

            double p = GetUniform();

            // Apply inverse of the Cauchy distribution function to a uniform
            return median + scale * Math.Tan(Math.PI * (p - 0.5));
        }

        public static double GetStudentT(double degreesOfFreedom)
        {
            if (degreesOfFreedom <= 0)
            {
                string msg = string.Format("Degrees of freedom must be positive. Received {0}.", degreesOfFreedom);
                throw new ArgumentException(msg);
            }

            // See Seminumerical Algorithms by Knuth
            double y1 = GetNormal();
            double y2 = GetChiSquare(degreesOfFreedom);
            return y1 / Math.Sqrt(y2 / degreesOfFreedom);
        }

        // The Laplace distribution is also known as the double exponential distribution.
        public static double GetLaplace(double mean, double scale)
        {
            double u = GetUniform();
            return (u < 0.5) ?
                mean + scale * Math.Log(2.0 * u) :
                mean - scale * Math.Log(2 * (1 - u));
        }

        public static double GetLogNormal(double mu, double sigma)
        {
            return Math.Exp(GetNormal(mu, sigma));
        }

        public static double GetBeta(double a, double b)
        {
            if (a <= 0.0 || b <= 0.0)
            {
                string msg = string.Format("Beta parameters must be positive. Received {0} and {1}.", a, b);
                throw new ArgumentOutOfRangeException(msg);
            }

            // There are more efficient methods for generating beta samples.
            // However such methods are a little more efficient and much more complicated.
            // For an explanation of why the following method works, see
            // http://www.johndcook.com/distribution_chart.html#gamma_beta

            double u = GetGamma(a, 1.0);
            double v = GetGamma(b, 1.0);
            return u / (u + v);
        }

    }
}
