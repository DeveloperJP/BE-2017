package bayesianclassifier;

import static bayesianclassifier.Bayesianclassifier.numberOfNodes;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import java.util.Random;
import java.util.Stack;
import weka.classifiers.bayes.net.ParentSet;

public class bat3 extends SearchAlgorithm
{
    static final long serialVersionUID = 6164792240778525312L;
    static int Parent;
    static int Edge;
    int jnew;
    protected boolean m_bInitAsNaiveBayes = true;    
    static double [][] frequency=new double[5][5];
    static int numberOfNodes;
    int maxedge;
    static int sum;
    int maxparent;
    int id;
    int[] A_i;
    boolean hasFound;
   

    public void setId(int id) {
        this.id = id;
    }
    
    public bat3() {
        this.frequency=Bayesianclassifier.frequency;
    } 

    public boolean hasCycles(BayesNet bayesNet) 
    {
        int numNodes =  bayesNet.getNrOfNodes();
        List[] adj = new ArrayList[numNodes];
        int[] colorNode = new int[numNodes];

        
        for(int i= 0; i<numNodes; i++)
        {
            adj[i] = new ArrayList<>();
        }
        
        //create the adj list
        
        for(int i= 0; i<numNodes; i++)
        {
            ParentSet parents = bayesNet.getParentSet(i);
            int[] parentList = parents.getParents();
            int parent_count = parents.getNrOfParents();
            
            for(int j=0; j<parent_count; j++)
            {
                adj[parentList[j]].add(i);
            }
        }

        for (int i=0; i<numNodes; i++)
        {
            colorNode[i]=0;
        }
        
        hasFound=false;
        
        for (int i=0; i<numNodes; i++)
        {
            if(colorNode[i]==0)
            {
                dfsVisit(adj, i, colorNode);
                if(hasFound==true)return true;
            }
            if(hasFound==true)return true;
        }
      
        return false;
    }
    
    
    private void dfsVisit(List[] adj, int i, int[] colorNode)
    {
        colorNode[i]=1;  // set the node color to gray
        List nodeAdjList = adj[i];
        
        for (int j=0; j<nodeAdjList.size(); j++)
        {
            int k=(int)adj[i].get(j);
            if (colorNode[k]==1)
            {
                hasFound=true;
                return;
            }
            if(colorNode[k]==0)
            {
                dfsVisit(adj, k, colorNode);
            }
        }
        colorNode[i]=2;
    }
    
   /* public static void initialization()throws Exception
    {
          frequency=Bayesianclassifier.frequency;
    
         for(int i=0; i<5;i++)
         {
             for(int j=0;j<5;j++)
             {
                 System.out.println("main matrix:::"+i+j+"   "+frequency[i][j]);
             }
         }
 
    
    }*/
    
    
    
    public static double[] frequencyValue(int Edge,int numberOfNodes)throws Exception
    {
        double[][] maxfrequency=new double[numberOfNodes][numberOfNodes]; 
        int ed=Edge;
        int n=numberOfNodes;
        int position[]=new int [2];
        int a=0;
        double positionNumber;
        double maxfreq[]=new double[Edge+Edge];
    
        for(int i=0; i<numberOfNodes;i++)                    //copy frequency array to the maxfrequency
        {
            for(int j=0;j<numberOfNodes;j++)
            {
                maxfrequency[i][j]=frequency[i][j];
            }
        }

        for(int e=0;e<ed;e++)                   //itrate till maximum edge
        {   
            double max=maxfrequency[0][0];
            for(int i1=0;i1<n;i1++)
            {
                for(int j1=0;j1<n;j1++)
                {
                    if(maxfrequency[i1][j1] > max)
                    {
                        max=maxfrequency[i1][j1];
                        position[0]=i1;
                        position[1]=j1;                               
                    }

                }
            }
            
            maxfrequency[position[0]][position[1]]=0;
            positionNumber=position[0]*10+position[1];
            maxfreq[a++]=max;
            maxfreq[a++]=positionNumber;
             
        } 
     
        return maxfreq;
   
    }
     
     
     
   public void setmaxedges(int edge)throws Exception
    {
        Edge=edge;
    }
     
    
    public void setmaxparents(int parent)throws Exception
    {
        Parent = parent;
    }
 
    public  int setparents()throws Exception
    {
       int parent=(int)(maxparent/2);
       return parent;
    }
  
  
    @Override
    public void buildStructure(BayesNet bayesNet, Instances instances) throws Exception
    {

        super.buildStructure(bayesNet,instances);

        Random ran = new Random();
        int numberOfNodes=bayesNet.getNrOfNodes();
        int parent=Parent;
        int edge=Edge;

     //   System.out.println("  Edge is..."+Edge);
      //   System.out.println("\n"+"  Parent is..."+Parent);
    //    parent = 1 + ran.nextInt(maxparent - 2 + 1);     //generate reandome number of parents using maximum number of parents
    //    edge = 1+ ran.nextInt(maxedge - 2 + 1);       //generate reandome number of edges using maximum number of edges
    //      
        //parent=receiveR_i();

           /* while(edge<=parent)
            {
               edge = 1+ ran.nextInt(edge- 2 + 1);
            }
            */
          // System.out.println("edge is :"+edge);
           // System.out.println("parent is :"+parent);

        double[] maxf=frequencyValue(edge,numberOfNodes);    //get maximum number of frequency of edges and their edges

             /*   for(int c=0;c<maxf.length;c++)
                 {
                     System.out.println("maxfrequency:::    "+maxf[c]);
                }
              */   
        double[] edgeweight=new double[edge];
        double[] pos=new double[edge];
        int s=0;

        for(int d=0;d<maxf.length;d++)                           //get only maximum number of edges from maxf array
        {
            edgeweight[s++]=maxf[d];
            d++;
        }
        /*
      for(int c=0;c<edgeweight.length;c++)
       {          
          System.out.println("("+c+")" +" maxfrequency:    "+edgeweight[c]+"\n");

       }
        */
        s=0;

        for(int d=1;d<maxf.length;d++)                           //get only position of edges to be added from maxf array
        {
            pos[s++]=maxf[d];
            d++;

        }

       // for(int c=0;c<pos.length;c++)
       //System.out.println("pos:    "+pos[c]);

            int x[] = new int[edge];                            
            int y[] = new int[edge];                            

                for(int z=0;z<pos.length;z++)                   //get one node from pos array
                {
                     x[z]=(int) (pos[z]/10);

                }

         //       for(int c=0;c<pos.length;c++)
          //      System.out.println("j is:    "+x[c]);

                for(int z=0;z<pos.length;z++)                   //get another node from pos array 
                {
                      y[z]=(int) (pos[z]%10);

                }

            //    for(int c=0;c<pos.length;c++)
             //   System.out.println("k is:    "+y[c]);

       Stack stc=new Stack();
       Stack stp=new Stack();


        //int v=0;
        for(int p=0;p<edge;p++)                                  //untill number of edges
       {                    


          int j=x[p];
          int k=y[p];


          if(stp.search(k)!=-1 && stc.search(j)!=-1)  
            continue;

            stp.push(k);
            stc.push(j);

            bayesNet.getParentSet((int) j).addParent((int) k, instances); 

           if(hasCycles( bayesNet))
               bayesNet.getParentSet((int) j).deleteParent((int) k, instances); 
          //   if(hashcycles(bayesNet,instances,j,k))
              // bayesNet.getParentSet((int) j).deleteParent((int) k, instances); 

            //  reversearc(bayesNet,instances,j,k);
                     //bayesNet.getParentSet(j).deleteParent(k, instances);


             // v++;

        }
      //  System.out.println("parents are:"+stp);
      //      System.out.println("child are:"+stc);
        //      System.out.println("new   child are:"+jnew);

    }   

    
    
}
        
        
     

    
    
