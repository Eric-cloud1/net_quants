using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace DemoQuants
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start");

            for (int i = 0; i < 50; i++)
            {
                Console.WriteLine("result {0}", Normal.GetNormal());
            }
          
           // checkStat stat = new checkStat();
           // stat.KSTest();
           // stat.TestDistributions();

            Console.ReadKey();

        }

      
        
        


    }
}
