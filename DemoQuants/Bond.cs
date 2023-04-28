using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DemoQuants
{
    public class Term
    {
        public Term() { }

        public double future_price(double S, double r, double time_to_maturity)
        {
            return Math.Exp(r * time_to_maturity)* S;
        }

        private  double term_structure_yield_from_discount_factor(
            double discountfactor,//
            double time)
        {
            return -Math.Log(discountfactor) / time;

        }

        private double term_structure_discount_factor_from_yield(
           double rate,//
           double time)
        {
            return Math.Exp(-rate / time);

        }
        private  double term_structure_foward_rate_from_yields(
           double discountfactor_t1,//
             double discountfactor_t2,//
           double time)
        {
            return Math.Log(discountfactor_t1 / discountfactor_t2) / time;

        }
        private  double term_structure_foward_rate_from_disc_facs(
           double discountfactor_t1,//
           double discountfactor_t2,//
           double t1,
           double t2)
        {
            return (discountfactor_t2 *(t2/ (t2 / t1)) - discountfactor_t1 *(t1 / t2));

        }

        private  double term_structure_yield_linearly_interpolate(
            double time,
            double[] obs_time,
            double[] obs_yield)
        {
            int number_obs = obs_time.Length;
            if (number_obs < 1) return 0;

            double t0 = obs_time[0];
            if (time < t0) return obs_yield[0];

            double tMax = obs_time[number_obs - 1];
            if (time >= tMax) return obs_yield[number_obs - 1];

            int t = 1;
            while ((t < number_obs) && (time > obs_time[t])) { t++; }

            double lambda = (obs_time[t] - time) / (obs_time[t] - obs_time[t - 1]);

            double r = obs_yield[t - 1] * lambda + obs_yield[t] * (1 - lambda);
            return r;

        }

        public  double fowardRate(double t1, double t2)
        {
            double dt1 = discountFactor(t1);
            double dt2 = discountFactor(t2);

            return term_structure_foward_rate_from_disc_facs(dt1, dt2, t1, t2);
        }

        public  double yield(double t)
        {
            return term_structure_yield_from_discount_factor(t, discountFactor(t));
        }

        public  double discountFactor(double t)
        {
            return term_structure_discount_factor_from_yield(yield(t), t);
        }

    }

    public class Bond
    {

        public Bond()
        {

        }

        public double bonds_price(double[] cashflow_t, double[] cashflow_amount, Term term)
        {
            double p = 0;
          
            for (int i = 0; i < cashflow_t.Length; i++)
            {
                p += term.discountFactor(cashflow_t[i]) * cashflow_amount[i];
            }
            return p;
        }
        public double bonds_duration(double[] cashflow_t, double[] cashflow_amount, Term term)
        {
            double s = 0;
            double d1 = 0;
            for (int i = 0; i < cashflow_t.Length; i++)
            {
                s += cashflow_amount[i] * term.discountFactor(cashflow_t[i]);
                d1 += cashflow_t[i] * cashflow_amount[i] * term.discountFactor(cashflow_t[i]);
            }

            return d1 / s;
        }
    }

    }

