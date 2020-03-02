
package bayesianclassifier;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Random;
import java.util.Stack;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.core.Instances;
import weka.gui.graphvisualizer.GraphVisualizer;


public class Bayesianclassifier
{ 
   
    static Stack qu=new Stack();
    static Stack gAi= new Stack();
    static Stack gRi= new Stack();

    static int numberOfNodes;  
    static double[][] frequency;
    static double MeanStuff[]=new double[9];
    static int maxedge;
    static int sum;
    static int maxparent;
    static double quality;
    static int pop_size = 10;
    static int A_i[] = new int[pop_size]; // max edges of bat i
    static int R_i[] = new int[pop_size]; // max parents of bat i
    static int max_iterations = 10;
    static int[] a = new int[pop_size]; 
    static double quality1,quality2,quality3;
    static double alpha=0.5, gema=2; 
    

     
   
    public static void quality(BayesNet bayesNet,Evaluation eval,Instances train)throws Exception{
  
         int i,count=10,fold=10;
         double[] arr= new double [fold];  
         double sum=0;
         for(i=0;i<count;i++)
         {    
             eval.crossValidateModel(bayesNet, train, 10,new Random(i));
             System.out.println(eval.toSummaryString("\nresults\n",true));
             System.out.println("correct%="+eval.pctCorrect());
             arr[i]=eval.pctCorrect();
             sum=sum+arr[i];
        }
      
        System.out.println("sum is :"+sum);
        quality=sum/10;
      
        double sum1=0,mean1;  
        System.out.println("quality : "+quality); //Display mean of all elements  
        double [] temp= new double[10];  
        for (i=0; i<10; i++) //calculate standard deviation  
        {  
          
            temp[i]=Math.pow((arr[i]-quality),2);  
            sum1+=temp[i];  
        }  
     
        mean1=sum1/9;  
        double deviation=Math.sqrt(mean1);  
        System.out.println("Deviation : "+ deviation);   
        double standard_error;
        standard_error=deviation/Math.sqrt(10);
        System.out.println("standard error"+standard_error);
        qu.push(quality);
        System.out.println("stack: "+qu);
               // }
        
     }
  
   public static void initializationFrequency(int TotalNode)throws Exception
   {    
       //intialise frequency in matrix(node*node)(global(for adding edges betwwen nodes))
       
         int n=TotalNode;
         double fmin=0,fmax=10;

         for(int i=0 ; i<n ; i++)
         {
             for(int j=0 ; j<n ; j++)
             {

                 if(i==j)
                 {
                     frequency[i][j]= 0 ;
                 }
                 else 
                 {
                     double b = Math.random();
                     frequency[i][j] = fmin + (fmax - fmin) * b;
                 }
             }
         }
         
         for(int i=0; i<numberOfNodes;i++)
         {
             for(int j=0;j<numberOfNodes;j++)
             {
                 System.out.println("main matrix:::"+i+j+"   "+frequency[i][j]);
             }
         }
   }
   
   public static int setmaxedges(int numberOfNodes)throws Exception
    {
        //set maximum possible edges
        
        maxedge=0;
        int a=numberOfNodes-1;
        
         for(int i=a;i>0;i--)
         {
            maxedge=i+maxedge; 
         }
        return maxedge; 
    }
   
     
    public static void main(String[] args) throws Exception 
    {
         //int i,count=10;
        Random ran = new Random();

        int id = 1;
        int ansp,anse,ParentRandom, EdgeRandom;
        //int maxedgee;
        
        BufferedReader breader=null;                   //for file reading
        breader= new BufferedReader(new FileReader("E:\\weka\\Weka-3-6\\data\\weather.nominal.arff")); 
        Instances train = new Instances (breader);
        train.setClassIndex(train.numAttributes() -1);
        breader.close();
         
        numberOfNodes=train.numAttributes();       //find number of attributes of the file
        Evaluation eval= new Evaluation(train);        //for data evaluation
         
        System.out.println("...n is "+numberOfNodes);
        
        int TotalNode = numberOfNodes;
        
        frequency=new double[numberOfNodes][numberOfNodes];
        
        initializationFrequency(TotalNode);        //function calling for intializing the frequency
      
       
         //(TODO:) initialize Ai and Ri
        for (int i=0; i<pop_size; i++)
        {
            // A_i[i] = i+1;
            // R_i [i]= (i+1) % (train.numAttributes() -1);
            
            // initialization od Ri..
            
            ParentRandom = 1 + ran.nextInt( 1000 - (numberOfNodes -1) + 1);                    
            ansp =  ParentRandom % (numberOfNodes -1) ;
            
            if(ansp == 0)
            {
                int m = 1 + ran.nextInt( (numberOfNodes -1) - 1  + 1); 
                ansp = ansp+m;   
                R_i[i]=ansp;                
            }
            else
            {
                R_i[i]=ansp;
            }
            
            
            // initialization of Ai..
                        
            maxedge=setmaxedges(numberOfNodes);

            EdgeRandom = 1 + ran.nextInt( maxedge - ( R_i[i] + 1 )  + 1); 
            anse =  EdgeRandom % numberOfNodes ;
                   
            A_i[i]=anse;

           while(A_i[i] < R_i[i])
            {
                int m = 1 + ran.nextInt( maxedge - ( R_i[i] + 1 )  + 1); 
                //ansp = ansp+m;   
                A_i[i]=m;                
            }
                       
        
        }
        

        
        for (int i=0; i<pop_size; i++)
        {
            System.out.println(" Initialized  Ri is : " + R_i[i] + " ans Ai is :  " + A_i[i] );
        }
        
        
        BayesNet BNCgbest = new BayesNet();  
        double Qgbest=0;
         
        bat3 bat[] = new bat3[pop_size]; 
        BayesNet bayes[] = new BayesNet[pop_size];
           
           
        //Initialize bat population
         
        for(int b=0;b<pop_size;b++)
        {
            
            bat[b]=new bat3();
            bat[b].setId(id);
            id++;
            //TODO: set algorithmically

            bat[b].setmaxedges(A_i[b]); 
            bat[b].setmaxparents(R_i[b]);
             
           // System.out.println("......" + R_i[b] + " ans  " + A_i[b] );
       
            bayes[b] = new BayesNet(); 
            bayes[b].setSearchAlgorithm(bat[b]);           
            bayes[b].buildClassifier(train);            //call buildclassifier from bat3 file
            quality(bayes[b],eval,train);               //call quality fuction
            
            if(quality > Qgbest)
            {
                BNCgbest=bayes[b];
                Qgbest=quality;
            }
            
            System.out.println("maximum quality(1):"+Qgbest);

            quality1=Qgbest;
        }
         
        //TODO
        
        //Initialize V_i for each edge;
        double V_i[][] = new double[numberOfNodes][numberOfNodes];
        for (int i = 0; i<numberOfNodes; i++){
            for (int j = 0; j <numberOfNodes; j++){
                V_i[i][j] = 1.0;
            }
        }
        
        // Loop till max_iterations
        int iter =0;
        // what..................................................................
        ParentSet par[]=new ParentSet [numberOfNodes];                  
        ParentSet Qpar[]=new ParentSet [numberOfNodes]; 
        int parents[][] = new int[numberOfNodes][numberOfNodes];
        int Qparents[][] = new int[numberOfNodes][numberOfNodes];
        
        int p[] = null;
        int Q[] = null;
        double dist ;
        double percentage;
        double alpha=0.5, gema=2;        

        while (iter < max_iterations)
        {
            iter++;
            //For each bat[b]
            for(int i =0; i<pop_size; i++)
            {
                //calucate the distance in quality from global best
                quality(bayes[i], new Evaluation(train), train);
                dist = Qgbest - quality;
                percentage=(dist)/10;


                System.out.println(" Distance is : "+percentage);
   
                for(int n=0;n<numberOfNodes;n++)
                {
                    par[n]=  bayes[i].getParentSet(n);
                    p=par[n].getParents();  
                    
                    for(int m=0;m<parents.length;m++)
                    {
//                                               System.out.println("p of mm "+ p[m]);
//                    System.out.println("......Parents of Node:" + n + "..............."+ p[m]);

                        int j=p[m];
                        parents[n][m]=j;
//                                            System.out.println("....parent[n][m]..."+ parents[n][m]);

                     }
                }
                
                
                
                    
                
                Stack flag = new Stack();

                for(int n=0;n<numberOfNodes;n++)
                {
                    Qpar[n]= BNCgbest.getParentSet(n);
                    Q=Qpar[n].getParents();
                      
                  

                    for(int m=0;m<parents.length;m++)
                    {
                        int k=Q[m];
                        Qparents[n][m]=k;
                    }
                }
               
                
               
                for(int n=0;n<numberOfNodes;n++)
                {
                    for(int m=0;m<parents.length;m++)
                    {
                        int u=Qparents[n][m];
                        int v=parents[n][m];
                       System.out.println("Qparent"+n+m+"="+ Qparents[n][m] + "\t Q frequency"+n+u+"="+ frequency[n][u] +"\tparent"+n+m+"="+ parents[n][m] + "\t Q frequency"+n+u+"="+ frequency[n][v] );
                    }
                }
                
              /*  for(int n=0;n<numberOfNodes;n++)
                {
                    for(int m=0;m<parents.length;m++)
                    {
                       System.out.println("parent"+n+m+"="+ parents[n][m]);
                    }
                }
                */
                int c=-1;
                if(dist>0)
                {
                    flag.clear();
                    System.out.println("*******add********");
                    for(int n=0;n<numberOfNodes;n++)
                    {
                        for(int m=0;m<Qparents.length;m++)
                        {
                            for(int x=0;x<numberOfNodes;x++)
                            {
                                for(int y= 0;y<parents.length;y++)
                                {
                                    if(n==x)
                                    {
                                        
                                        int f1= (n*10)+ Qparents[n][m];
                                        int f = flag.search( f1);
                                       // System.out.println("...search f1 " +f1);

                                        if( f == -1)
                                        {
                                            if(Qparents[n][m]==parents[x][y])
                                            {
                                                 int k=Qparents[n][m];
                                                 c= (n*10)+k;

                                                 if(n == 0  && k == 0)
                                                 {
                                                    frequency[n][k]=0;

                                                 }
                                                 else
                                                 {
                                                     if(k!=0)
                                                     {
                                                        System.out.println("---------"+frequency[n][k]+" n  is-- "+n+"  k is-- "+k);
                                                        frequency[n][k]= percentage + frequency[n][k];
                                                        System.out.println("********************************************************new************"+frequency[n][k]);
                                                        flag.push(c);
                                                        System.out.println(" c is "+c);

                                                     }

                                                 }
                                            }  
                                        }
                                    }
                                }
                            }
                        }
                    }
                }        
               
                else if(dist<0)
                {
                    flag.clear();

                    System.out.println("*******sub********");
                    for(int n=0;n<numberOfNodes;n++)
                    {
                        for(int m=0;m<Qparents.length;m++)
                        {
                            for(int x=0;x<numberOfNodes;x++)
                            {
                                for(int y= 0;y<parents.length;y++)
                                {
                                    if(n==x)
                                    {
                                         int f1= (n*10)+ Qparents[n][m];
                                        int f = flag.search( f1);
                                       // System.out.println("...search f1 " +f1);
                                        if( f == -1)
                                            
                                        {
                                            if(Qparents[n][m]==parents[x][y])
                                            {
                                                 int k=Qparents[n][m];
                                                 c= (n*10)+k;
                                                 if(n == 0  && k == 0)
                                                 {
                                                    frequency[n][k]=0;

                                                 }
                                                  else
                                                 {
                                                     if(k!=0)
                                                     {
                                                        System.out.println("---------"+frequency[n][k]+" n  is-- "+n+"  k is-- "+k);
                                                        frequency[n][k]= percentage - frequency[n][k];
                                                        System.out.println("********************************************************new************"+frequency[n][k]);
                                                        flag.push(c);
                                                        System.out.println(" c is "+c);

                                                     }
                                                 }
                                            } 
                                        }
                                    }
                                }
                            }
                        }
                    }
                }     
               
                //use equation 3 to update the frequencies;
                for(int l=0; l<numberOfNodes;l++)
                {
                    for(int j=0;j<numberOfNodes;j++)
                    {
                        System.out.println("updated  matrix is ::"+l+j+"   "+frequency[l][j]);
                    }
                }
               
                
                
                //generate new solutions for each bat[b] using new frequency table.
                
                gAi.clear();
                gRi.clear();
                
                int ai=1;
                for(int b=0;b<pop_size;b++)
                {
                    bat[b]=new bat3();
                    bat[b].setId(id);
                    id++;
                    //TODO: set algorithmically
                    bat[b].setmaxedges(A_i[b]); 
                    bat[b].setmaxparents(R_i[b]);


                    bayes[b] = new BayesNet(); 
                    bayes[b].setSearchAlgorithm(bat[b]);           
                    bayes[b].buildClassifier(train);            //call buildclassifier from bat3 file
                    quality(bayes[b],eval,train);               //call quality fuction

                   // find new global best

                    if(quality > Qgbest)
                    {
                        BNCgbest=bayes[b];
                        int tempAi=A_i[b];
                      //  BNCgbest.getmaxParents();
                        
                        int tempRi=R_i[b];
                        gAi.push(tempAi);
                        gRi.push(tempRi);
                       // AIth[ai++]=b;
                        Qgbest=quality;
                    }

                    else
                    {
                        BNCgbest=bayes[b];
                        int tempAi=A_i[b];
                         int tempRi=R_i[b];
                        gAi.push(tempAi);
                        gRi.push(tempRi);
                       
                        Qgbest=quality;

                    }
                    System.out.println("maximum quality(2):"+Qgbest);
                    System.out.println("Ai stack is :"+gAi);
                                           
                    /* for(int j=0;j<AIth.length;j++)
                    
                    {
                        System.out.println(" AIth is "+AIth[j]);
                    
                    }*/
                    
                               quality2=Qgbest;

                }
                       
            }
            
                // find new global best
                
                
                // Start Local search: (A_i and R_i are updated for each bat)
                
                
                int tempA[]=new int[1];

                tempA[0] = (int) gAi.pop();
                
                int tempR[]=new int[1];

                tempR[0] = (int) gRi.pop();
                
                for(int i =0; i<pop_size; i++)
                {
                    quality(bayes[i], new Evaluation(train), train);
                    dist = Qgbest - quality;
                    percentage=dist/100;
                
                                               
                    System.out.println(" Distance is : "+dist);
                 

                    
                     System.out.println("......................................................."+gAi);

                    
                    
                    System.out.println("......................................................."+gRi);


                    System.out.println(".......................................temp is..."+tempA[0] + "......................."+A_i[i]);

                    System.out.println(".......................................temp is..."+tempR[0] + "......................."+R_i[i]);

                   
                    // equation for update  A_i and R_i 
                    
                    
                    /*
                    R_i[i] = (int) (R_i[0] * y);
                    
                    while(R_i[i] > maxparent)
                    {
                        R_i[i] = 1 + ran.nextInt( maxparent - 1  + 1); 

                    }
                    System.out.println("........ri  ...."+R_i[i]);
                    
                    */
                    if(tempA[0] == A_i[i])
                    {
                        if(dist > 0)
                        {
                            A_i[i] = (int) (A_i[i] * ( 1 + alpha) );

                            while(A_i[i] > maxedge && A_i[i] == 0)
                            {
                                A_i[i] = 1 + ran.nextInt( maxedge - 5  + 1); 
                            }
                        }        

                        else if(dist<0)
                        {
                            System.out.println("*******sub********");

                            A_i[i] = (int) (A_i[i] * ( 1 - alpha));

                            if(A_i[i] > maxedge)
                            {
                                A_i[i] = 1 + ran.nextInt( maxedge - 5  + 1); 
                            }
                        }
                        
                    }  
                    
                    if(tempR[0] == R_i[i])
                    {
                      R_i[i] = (int) (R_i[0] * (1 - Math.exp( - ( gema * i))));                        
                    } 
                    
                    
                     
                    
                }
                            System.out.println(".................................................. +------------------------------" );

                     
                for (int i=0; i<pop_size; i++)
                {
                    System.out.println(" new  Ri is : " + R_i[i] + " ans Ai is :  " + A_i[i] );
                }
       

                // quality checking after update A_i and R_i

                
                for(int b=0;b<pop_size;b++)
                {
                    bat[b]=new bat3();
                    bat[b].setId(id);
                    id++;
                    //TODO: set algorithmically
                    bat[b].setmaxedges(A_i[b]); 
                    bat[b].setmaxparents(R_i[b]);


                    bayes[b] = new BayesNet(); 
                    bayes[b].setSearchAlgorithm(bat[b]);           
                    bayes[b].buildClassifier(train);            //call buildclassifier from bat3 file
                    quality(bayes[b],eval,train);               //call quality fuction

                   // find new global best

                    if(quality > Qgbest)
                    {
                        BNCgbest=bayes[b];

                        Qgbest=quality;
                    }

                   System.out.println("maximum quality(3):"+Qgbest);

                               quality3=Qgbest;

                }
                
        
    
                
                
                
                //past alpha=0.1
                
                    //generate new bats using new values of A_i and R_i
                    //find global best
                
            
         // End Loop  
         
        }
         displayBayesianGraph(BNCgbest);
                
                
        
             System.out.println("maximum quality(1):"+quality1+"\n max quality(2):"+quality2+"\n max quality(3):"+quality3);
                       //  System.out.println("gAi stack is :"+gAi);
            /*for(int j=0;j<AIth.length;j++)
                    {
                        System.out.println(" AIth is "+AIth[j]);
                    
                    }
                    */

        
   
    }
    
    
    
    
    private static void displayBayesianGraph(BayesNet bayesNet) throws FileNotFoundException
    {
        String s=new String(bayesNet.toXMLBIF03());
        final javax.swing.JFrame jf= new javax.swing.JFrame("Weka Classifier Tree Visualizer:HGBN");
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        GraphVisualizer graph=new GraphVisualizer();
        try{
            graph.readBIF(s);
        } catch(Exception ex){
//            Logger.getLogger(Bat.class.getName()).log(Level.SEVERE, null,ex);
        }jf.getContentPane().add(graph,BorderLayout.CENTER);
         jf.addWindowListener(new java.awt.event.WindowAdapter() {
             public void windowClosing(java.awt.event.WindowEvent e){
                 jf.dispose();
             }
            });
        
      
      jf.setVisible(true);
      PrintWriter writer= new PrintWriter("graph"); 
      writer.print(s);
      writer.close();
    }
}

