using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DemoQuants
{
    public class Option
    {
        public Option() { }

        //At maturity, a call option is worth Max(S – X) and a put is worth Max(X – S)
        //http://finance-old.bi.no/~bernt/gcc_prog/recipes/recipes/node10.html   
        //Monte carlo simulation

        public double Call_european_montecarlo(
                        double S, //price of the asset/
                        double X, //exercise price
                        double r, //risk free rate
                        double sigma, //volatility of the asset
                        double time, //time to maturity in year
                        long no_sims, //number simulation
                        Normal normal
                        )
        {
            double R = (r - 0.5 * Math.Pow(sigma, 2)) * time;
            double SD = sigma * Math.Sqrt(time);
            double simulated_payoff = 0.0;
            double x =0.0;
            for (long n = 1; n <= no_sims; n++)
            {
                x = normal.random_normal();
                double price_Time = S * Math.Exp(R + SD * x);
                simulated_payoff += Math.Max(0.0, price_Time - X);
            }
            return Math.Exp(-r * time) * (simulated_payoff / no_sims);
        }



        public double Put_price_bermuda_binomial(double S, // spot price
        double X, // exercise price  
        double r, double q, //interest & countinuous payout
        double sigma, double time,  // volatility //time to maturity
        Double potential_exercise_t, //exercise
        int steps //number in binomial tree
        )
        {
            double delta_t = time / steps;
            double R = Math.Exp(r * delta_t);// interest at each steps
            double Rinv = 1 / R;
            double u = Math.Exp(sigma * Math.Sqrt(delta_t)); // up moves
            double uu = u * u; 
            double d = 1 / u;
            double p_up = (Math.Exp((r-q)*delta_t)-d)/(u - d);
            double p_down = 1 - p_up;
            Double prices = new Double();
            Double put_values = new Double();
            Double potential_exercise_steps = new Double();                  
            for (int i = 0; i < potential_exercise_t.Count; i++)
            {
                double t = potential_exercise_t[i];
                if ((t > 0) && (t < time))
                {
                    int j = potential_exercise_steps.Count;
                    potential_exercise_steps[j] = t / delta_t;
                }
            }
            prices[0] = S * Math.Pow(d, steps);
            for (int i = 1; i <= steps; i++) prices[i] = uu * prices[i - 1];
            for (int i = 0; i <= steps; i++) put_values[i] = Math.Max(0, (X - prices[i]));
            bool check_exercise_this_step = false;
            for (int step = steps - 1; step >= 0; step--)
            {
                for (int j = 0; j < potential_exercise_steps.Count; j++)
                {
                    if(step == potential_exercise_steps[j]){check_exercise_this_step = true;}
                }
            }
            for (int i = 0; i <= steps; i++)
            {
                put_values[i] = (p_up * put_values[i + 1] + p_down * put_values[i]) * Rinv;
                prices[i] = d * prices[i + 1];
                if (check_exercise_this_step== true) put_values[i] = Math.Max(put_values[i], X - prices[i]);
            }
            return put_values[0];
        }


        public double Call_asian_geometrix_average(double S, double X, double r, 
            double q, double sigma, double time, Normal n)
        {

        double sigma_sqr = Math.Pow(sigma, 2);
        double adj_div_yield = 0.5 * (r + q + sigma_sqr);
        double adj_sigma = sigma / Math.Sqrt(3);
        double adj_sigma_sqr = Math.Pow(adj_sigma,2);
        double time_sqr = Math.Pow(adj_sigma, 2);
        double d1_1 = Math.Log(S / X) + (r - adj_div_yield + 0.5 * adj_sigma_sqr) * time;
        double d1 = d1_1 / (adj_sigma * time_sqr);
        double d2 = d1 - (adj_sigma * time_sqr);
        double call_price = S * Math.Exp(-adj_div_yield * time) * n.N(d1) - X * Math.Exp(-r * time) * n.N(d2);

        return call_price;
        }


        public double Call_future_american_binomial(double F, // price futures contract
            double X, double r, double sigma, double time, int no_steps)
        {
            Double futures_prices = new Double();
            Double call_values = new Double();
            double delta_t = time / no_steps; // ticks
            double Rinv = Math.Exp(-r* delta_t);
            double u = Math.Exp(sigma * Math.Sqrt(delta_t));
            double d =1/u;
            double uu = u * u;
            double pUp = (1 - d) / (u - d); // probability
            double pDown = 1 - pUp;

            futures_prices[0] = F * Math.Pow(d, no_steps);

            for (int i = 1; i <= no_steps; i++) futures_prices[i] = uu * futures_prices[i - 1]; // terminal nodes
            for (int i = 1; i <= no_steps; i++)  call_values[i] = Math.Max(0,(futures_prices[i]-X));
            for (int step = no_steps - 1; step >= 0; step--)
            {
                for (int i = 0; i <= step; i++)
                {
                    futures_prices[i] = d * futures_prices[i + 1];
                    call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv;
                    call_values[i] = Math.Max(call_values[i], (futures_prices[i] - X)); //check for exercise
                }
            }
            return call_values[0];
        }


        public double Call_currency_american_binomial(double S, 
            double X, double r, double r_f, double sigma, double time, int no_steps)
        {
            Double exchange_rates = new Double();
            Double call_values = new Double();
            double delta_t = time / no_steps; // ticks
            double Rinv = Math.Exp(-r * delta_t);
            double u = Math.Exp(sigma * Math.Sqrt(delta_t));
            double d = 1 / u;
            double uu = u * u;
            double pUp = Math.Exp(((r - r_f)* delta_t)- d)/ (u-d) ; // adjust for foreign int rate
            double pDown = 1 - pUp;
            exchange_rates[0] = S * Math.Pow(d,no_steps);
            for(int i = 1; i<= no_steps; i++)
            {
                exchange_rates[i] = uu * exchange_rates[i-1]; // terminal nodes
            }
            for(int i = 0; i<= no_steps; i++) call_values[i] = Math.Max(0, exchange_rates[i]-X);
            for(int step = no_steps - 1; step >=0; step--)
            {
                for (int i = 0; i <= step; i++)
                {
                    exchange_rates[i] = d* exchange_rates[i+1];
                    call_values[i] = (pDown * call_values[i] + pUp * call_values[i+1])* Rinv;
                    call_values[i] = Math.Max(call_values[i], exchange_rates[i] - X); //check for exercise
                }
            }
            return call_values[0];
        }

       
        public double Call_warrant_adjusted_blackscholes(Normal nl, double S, double K, //strike price
            double r, double sigma, double time, 
            double m, double n// number of warrants & number of shares outstanding
            )
        {

            double epsilon = 0.00001;
            double time_sqrt = Math.Sqrt(time);
            double w = (n /(n+m))*Call_black_scholes(S,K,r,sigma,time, new Normal());
            double g = w - (n / (n + m)) * Call_black_scholes(S + (m + n) * w, K, r, sigma, time, new Normal());
            while (Math.Abs(g) > epsilon)
            {
                double d1_1 = Math.Log((S + (m / n)) / K) + r * time;
                double d1 = d1_1 / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt;
                double gprime = 1 - (m/n) * nl.N(d1);
                w = w - g / gprime;
                g = w - (n / (n + m)) * Call_black_scholes(S + (m / n) * w, K, r, sigma, time, new Normal());
            }
            return w;
        }

        private double Call_european_payout(double Si, double X, double r, double sigma, double time)
        {
            return 0;
        }

   
        public double Call_aproximated_baw(double S, double X, double r, 
            double b, double sigma, double time, Normal nl)
        {
            double accuracy = Math.Pow(10, -6);
            double sigma_sqr = sigma * sigma;
            double time_sqrt = Math.Sqrt(time);
            double nn = 2 * b / sigma_sqr;
            double m = 2 * r / sigma_sqr;
            double K = 1 - Math.Exp(-r * time);
            double q2 = (-(nn - 1) + Math.Sqrt(Math.Pow((nn - 1), 2) + (4 * m / K))) * 0.5;

            double q2_inf = 0.5 * (-(nn - 1) + Math.Sqrt(Math.Pow((nn - 1), 2) + 4 * m));
            double S_star_inf = X / (1 - 1 / q2_inf);
            double h2 = -(b * time + 2 * sigma * time_sqrt) * (X / (S_star_inf - X));
            double S_seed = X + (S_star_inf - X) * (1 - Math.Exp(h2));

            int no_iterations = 0;
            double Si = S_seed;
            double g = 1, gprime = 1, c = 0;
            while ((Math.Abs(g) > accuracy)&&(Math.Abs(gprime) > accuracy)
                &&(no_iterations++ < 500)&&(Si > 0)){

                 c = Call_european_payout(Si,X,r,sigma,time);

                double d1 = (Math.Log(Si/X) + (b=0.5*sigma_sqr)* time)/(sigma*time_sqrt);
                g = (1-1/q2)*Si-X-c+(1/q2)*Si*Math.Exp((b-r)*time)* nl.N(d1);

                gprime =(1-1/q2)*(1-Math.Exp((b-r)*time)*nl.N(d1))
                    + (1/q2)*Math.Exp((b-r)*time)*nl.n(d1)*(1/(sigma*time_sqrt));
                Si = Si - (g/gprime);
            }
            double S_star = 0;
            if(Math.Abs(g) > accuracy){S_star = S_seed;}
            else{S_star = Si;};

            double C =0;
             c = Call_european_payout(Si,X,r,sigma,time);
            if(S>=S_star){
                C = S -X;
            }
            else {
                double d1 = (Math.Log(S_star /X)+(b+0.5*sigma_sqr)*time)/(sigma*time_sqrt);
                double A2 = (1 - Math.Exp((b-r)*time)*nl.N(d1))*(S_star/q2);
                C = c + A2*Math.Pow((S/S_star),q2);
            }
            return Math.Max(C,c);
        }

        public double Call_black_scholes(double S, double X, double r, double sigma, double time, Normal nl)
        {
            double time_sqrt = Math.Sqrt(time);
            double d1 = (Math.Log(S / X) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt;
            double d2 = d1 - (sigma * time_sqrt);
            double c = S * nl.N(d1) - X * Math.Exp(-r * time) * nl.N(d2);
            return c;
        }

        public double Call_black_scholes_partial_derivatives(double S, double X 
            // strike (exercise price)
            , double r, double sigma, double time, Normal nl,
            out double Delta, // Delta sensitivity of the option price relative to changes in the price 
            out double Gamma, // how the delta of an option will change relative to a 1 point move.
            out double Theta, // Theta shows how much value the option price over time
            out double Vega, // Theta shows how much value the option price over sigma
            out double Rho // Rho shows how much value the option price over r
            )
        {
            double time_sqrt = Math.Sqrt(time);
            double d1 = (Math.Log(S / X) + r * time) / (sigma * time_sqrt) + 0.5 * sigma * time_sqrt;
            double d2 = d1 - (sigma * time_sqrt);
            double c = S * nl.N(d1) - X * Math.Exp(-r * time) * nl.N(d2);

            Delta = nl.N(d1);
            Gamma = nl.n(d1) / (S * sigma * time_sqrt);
            Theta = -(S * sigma * nl.n(d1)) / (2 * time_sqrt) - r * X * Math.Exp(-r * time) * nl.N(d2);
            Vega = S * time_sqrt * nl.n(d1);
            Rho = X * time * Math.Exp(-r * time) * nl.N(d2);

            return c;

        }


        public double Call_black_scholes_implied_volatility(double S, double X, 
            double r,  double time, double option_price, Normal nl)
        {
            double sigma_low = 0.0001;
            double price = Call_black_scholes(S, X, r, sigma_low, time, new Normal());
            if (price > option_price) return 0;

            double accuracy = Math.Pow(10, -5);
            int max_iterations = 100;
            double high_value = Math.Pow(10, 10);
            double error = Math.Pow(10, -40);
            double sigma = 0, test = 0;
            double sigma_high = 0.5;

            price = Call_black_scholes(S, X, r, sigma_high, time, new Normal());
            while (price < option_price)
            {
                sigma_high = 2 * sigma_high;
                price = Call_black_scholes(S, X, r, sigma_high, time, new Normal());
                if (sigma_high > high_value) return error;
            }

            for (int i = 0; i < max_iterations; i++)
            {
                sigma = (sigma_low + sigma_high) * 0.5;
                price = Call_black_scholes(S, X, r, sigma, time, new Normal());
                test = (price - option_price);
                if(Math.Abs(test) < accuracy) return sigma;

                if(test < 0) sigma_low = sigma;
                
                else sigma_high = sigma;

            }
            return error;
            
        }
        public double Call_merton_jump_diffusion(double S, double X, double r,
            double sigma, double time_to_maturity, Normal nl, double lambda, double kappa, double delta)
        {
            int max_iterations = 50;
            double sigma_n = 0, r_n =0,  log_n = 0;
            double tau = time_to_maturity;
            double sigma_sqr = sigma * sigma;
            double delta_sqr = delta * delta;
            double lambdaprime = lambda * (1 + kappa);
            double gamma = Math.Log(1 + kappa);

            double c = Math.Exp(-lambdaprime * tau) 
                * Call_black_scholes(S, X, r - lambda * kappa, sigma, tau, new Normal());
            
            for (int n = 1; n < max_iterations; n++)
            {
                log_n += Math.Log((double)n);
                sigma_n = Math.Sqrt(sigma_sqr + n * delta_sqr / tau);
                r_n = r - lambda * kappa * n * gamma / tau;

                c += Math.Exp(-lambdaprime * tau + n * Math.Log(lambdaprime * tau) - log_n) 
                    * Call_black_scholes(S, X, r_n, sigma_n, tau, new Normal());

            }
            return c;
        }

        public double Call_european_payout(double S, double X, double r, double q, // yield on underlying
            double sigma, double time, Normal n)
        {

            double sigma_sqr = Math.Pow(sigma, 2);
            double time_sqrt = Math.Sqrt(time);
            double d1 = (Math.Log(S / X) + (r - q + 0.5 * sigma_sqr) * time) / (sigma * time_sqrt);
            double d2 = d1 - (sigma * time_sqrt);
            double call_price = S * Math.Exp(-q * time) * n.N(d1) - X * Math.Exp(-r * time) * n.N(d2);
            return call_price;
        }

        public double Call_european_dividends(double S, double X, double r, double sigma,
            double time_to_maturity, Double dividend_times, Double dividend_amounts)
        {
            double adjusted_S = S;
            for (int i = 0; i <= dividend_times.Count; i++)
            {
                if (dividend_times[i] <= time_to_maturity) 
                { adjusted_S -= dividend_amounts[i] * Math.Exp(-r * dividend_times[i]); }

            }
            return Call_black_scholes(adjusted_S, X, r, sigma, time_to_maturity,new Normal());

        }

        public double Call_european_binomial(double S, double X, double r, double sigma, double t, int steps)
        {
            double R = Math.Exp(r*(t/steps));
            double Rinv = 1/ R;
            double u = Math.Exp(sigma * Math.Sqrt(t/steps)); //up moves
            double uu = u*u;
            double d = 1/u;
            double p_up = (R - d) / (u - d);
            double p_down = 1 - p_up;
            Double prices = new Double();
            Double call_values = new Double();

            prices[0] = S * Math.Pow(d, steps);
            for (int i = 1; i <= steps; i++) prices[i] = uu * prices[i - 1];
            for (int step = steps - 1; step >= 0; step--)
            {
                for (int i = 0; i <= step; i++) 
                { call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv; }
            }
            return call_values[0];
        }


        public double Call_american_binomial(double S, double X, double r, double sigma, double t, int steps)
        {
            double R = Math.Exp(r * (t / steps));
            double Rinv = 1 / R;
            double u = Math.Exp(sigma * Math.Sqrt(t / steps)); //up moves
            double uu = u * u;
            double d = 1 / u;
            double p_up = (R - d) / (u - d);
            double p_down = 1 - p_up;
            Double prices = new Double();
            Double call_values = new Double();

            prices[0] = S * Math.Pow(d, steps);
            for (int i = 1; i <= steps; i++) prices[i] = uu * prices[i - 1];
            for (int i = 0; i <= steps; i++) call_values[i] = Math.Max(0, (prices[i] - X));


            for (int step = steps - 1; step >= 0; step--)
            {
                for (int i = 0; i <= step; i++)
                { 
                    call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv;
                    prices[i] = d * prices[i + 1];
                    call_values[i] = Math.Max(call_values[i], prices[i] - X); //check for exercise

                }
            }
            return call_values[0];
        }


        public double Call_american_binomial_partial_derivatives
            (double S, double X, double r, double sigma, double t, int steps,
            out double Delta, // Delta sensitivity of the option price relative to changes in the price 
            out double Gamma, // how the delta of an option will change relative to a 1 point move.
            out double Theta, // Theta shows how much value the option price over time
            out double Vega, // Theta shows how much value the option price over sigma
            out double Rho // Rho shows how much value the option price over r
            )
        {
            double R = Math.Exp(r * (t / steps));
            double delta_t = t / steps;
            double Rinv = 1 / R;
            double u = Math.Exp(sigma * Math.Sqrt(t / steps)); //up moves
            double uu = u * u;
            double d = 1 / u;
            double p_up = (R - d) / (u - d);
            double p_down = 1 - p_up;
            Double prices = new Double();
            Double call_values = new Double();

            prices[0] = S * Math.Pow(d, steps);
            for (int i = 1; i <= steps; i++) prices[i] = uu * prices[i - 1];
            for (int i = 0; i <= steps; i++) call_values[i] = Math.Max(0, (prices[i] - X));


            for (int step = steps - 1; step >= 0; step--)
            {
                for (int i = 0; i <= step; i++)
                {
                    call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv;
                    prices[i] = d * prices[i + 1];
                    call_values[i] = Math.Max(call_values[i], prices[i] - X); //check for exercise

                }
            }

            Delta = (call_values[1] - call_values[0]) / (S * u - S * d);
            double h = 0.5 * S * (uu - d * d);
            Gamma = ((call_values[2] - call_values[1])/(S*(uu-1))-(call_values[1]-call_values[0])/(S*(1-d*d)))/h;
            Theta = (call_values[1] - call_values[0]) / (2 * delta_t);
            double diff = 0.05;
            double tmp_sigma = sigma + diff;
            double tmp_prices = Call_american_binomial(S, X, r, tmp_sigma, t, steps);
            Vega = (tmp_prices - call_values[0]) / diff;
            diff = 0.05;
            double tmp_r = r + diff;
            tmp_prices = Call_american_binomial(S, X, tmp_r, sigma, t, steps);
            Rho = (tmp_prices - call_values[0]) / diff;

            return call_values[0];
        }

        double Call_american_binomial_dividend(double S, double X, double r, double sigma,
            double tau, double D1, double tau1, Normal n)
        {
            if (D1 <= X * (1 - Math.Exp(-r * (tau - tau1)))) //check for exercise
                return Call_black_scholes(S - Math.Exp(-r * tau1) * D1, X, r, sigma, tau, new Normal());

            double accuracy = Math.Pow(10, -6);
            double sigma_sqr = sigma * sigma;
            double tau_sqrt = Math.Sqrt(tau);
            double tau1_sqrt = Math.Sqrt(tau1);
            double rho = -Math.Sqrt(tau1);

            double S_bar = 0;
            double S_low = 0;
            double S_high = S;
            double c = Call_black_scholes(S_high, X, r, sigma, tau - tau1, n);
            double test = c - S_high - D1 + X;
            while((test > 0)&&(S_high <= Math.Pow(10, 10))){
                S_high *= 2;
                c = Call_black_scholes(S_high, X, r, sigma, tau - tau1, n);
                test = c - S_high - D1 +X;
            }
            if(S_high > Math.Pow(10,10)) return Call_black_scholes(S_high,X,r,sigma, tau - tau1, n);

            S_bar = 0.5 * S_high;
            c = Call_black_scholes(S_bar, X,r, sigma, tau-tau1, n);
            test = c - S_high - D1 +X;
            while (( Math.Abs(test) > accuracy)&&((S_high - S_low)> accuracy)){
                if (test < 0) S_high = S_bar;
                else S_low = S_bar;
                S_bar = 0.5 * (S_high + S_low);
                c = Call_black_scholes(S_bar, X, r, sigma, tau - tau1, n);
                test = c - S_bar - D1 + X;
            }

            double a1 = (Math.Log((S - D1 * Math.Exp(-r * tau1)) / X) + (r + 0.5 * sigma_sqr) * tau) / (sigma * tau_sqrt);
            double a2 = a1 - sigma * tau_sqrt;
            double b1 = (Math.Log((S - D1 * Math.Exp(-r * tau1)) / S_bar) + (r + 0.5 * sigma_sqr) * tau1) / (sigma * tau1_sqrt);
            double b2 = b1 - sigma * tau1_sqrt;
            double C = (S - D1 * Math.Exp(-r * tau1)) * n.N(b1) + 
                (S - D1 * Math.Exp(-r * tau1)) * n.N(a1, -b1, rho) - 
                (X * Math.Exp(-r * tau)) * n.N(a2, -b2, rho) - (X - D1) * Math.Exp(-r * tau1) * n.N(b2);


            return C;
        }

        public double Call_american_discrete_dividends_binomial(double S, double X, double r, double sigma,
            double t, int steps, Double dividend_times, Double dividend_amounts)
        {
            int steps_before_dividend = (int)(dividend_times[0] / t * steps);
            int no_dividends = dividend_times.Count;
            if (no_dividends == 0) return Call_american_binomial(S, X, r, sigma, t, steps);

            double R = Math.Exp(sigma * Math.Sqrt(t / steps));
            double Rinv = 1 / R;
            double u = Math.Exp(sigma * Math.Sqrt(t / steps));
            double d = 1 / u;
            double pUp = (R / d) / (u - d);
            double pDown = 1 - pUp;
            double dividend_amount = dividend_amounts[0];
            Double temp_dividend_times = new Double();
            Double temp_dividend_amounts = new Double();

            for (int i = 0; i < no_dividends - 1; i++)
            {
                temp_dividend_amounts[i] = dividend_amounts[i + 1];
                temp_dividend_times[i] = dividend_times[i + 1] - dividend_times[0];
            }

            Double prices = new Double();
            Double call_values = new Double();
            prices[0] = S * Math.Pow(d, steps_before_dividend);
            for (int i = 1; i <= steps_before_dividend; i++) prices[i] = u * u * prices[i - 1];

            for (int i = 1; i <= steps_before_dividend; i++)
            {
                double value_alive =
                    Call_american_discrete_dividends_binomial(prices[i] - dividend_amount, X, r, sigma,
                    temp_dividend_times[0], steps - steps_before_dividend,
                    temp_dividend_times, temp_dividend_amounts);

                call_values[i] = Math.Max(value_alive, (prices[i] - X)); //compare to exercise now
            }

            for (int step = steps_before_dividend - 1; step >= 0; step--)
            {
                for (int i = 0; i <= step; i++)
                {
                    prices[i] = d * prices[i + 1];
                    call_values[i] = (pDown * call_values[i] + pUp * call_values[i + 1]) * Rinv;
                    call_values[i] = Math.Max(call_values[i], prices[i] - X);
                }
            }
            return call_values[0];
        }

    }
}
