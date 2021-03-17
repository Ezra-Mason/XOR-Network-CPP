#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <time.h>
#include "functions.hpp"

//######################################################################
//This is a program for a three layer, fully connected & feed forward 
//neural network with one hidden layer of 3 nodes. Designed to learn an 
//XOR gate using the Metroplis Algorithm to learn (minimise global 
//error).
//######################################################################
int main(){
    //inverse temperature 
    double beta =0.1;
    //All the networks weights
    //   0,1,2,3->bais,  4..12->weights into hidden nodes,   13,15->hidden to output weights
    // [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  ]
    // [ w40 w50 w60 w70 w41 w42 w43 w51 w52 w53 w61 w62 w63 w74 w75 w76 ]
    std::vector<double> current_weights(16);
    std::vector<double> new_weights(16);
    //truth table for a 3 input XOR gate
    int truth_table[8][4] = {{0,0,0,0},
                             {0,0,1,1}, 
                             {0,1,0,1}, 
                             {0,1,1,0}, 
                             {1,0,0,1}, 
                             {1,0,1,0}, 
                             {1,1,0,0}, 
                             {1,1,1,1}};
    //Welcome message
    std::cout<<"Welcome to XOR-gate neural network by Ezra Mason!\n";

    //Initialise all the weights randomly
    current_weights = SetUpWeights(current_weights);
    std::cout<<"\n Randomly assigned weights: \n";
    OutputWeights(current_weights);

    //Initialise random seed 
    srand(time(NULL));

    int n_max = 1000000;
    std::cout<<"\n Starting loop... \n";
    for (int i =0; i<n_max; i++){
        // inverse temperature increases over time, simulating the system cooling
        beta = beta + 1000.0/n_max;

        //make a copy of the current
        new_weights = current_weights;

        //generate a number between 0 and 15 to pick a random weight
        //change the selected weight by a random value between -10 and 10
        unsigned short int random_index = ( double(rand()) / (double(RAND_MAX)) ) * current_weights.size();
        new_weights[random_index] =(double(rand()) / (double(RAND_MAX)) - 0.5) * 20; 

        //Get the global errors if the two weight configurations
        double E_current = GlobalError(current_weights, truth_table);
        double E_new = GlobalError(new_weights, truth_table);

        //decide which configuration we want to keep 
        if(E_new-E_current<0.0){
            current_weights = new_weights;
        }
        else{
            double p = exp((E_current-E_new)*beta);
            double r = double(rand()) / (double (RAND_MAX) );
            if(r<p){
                current_weights = new_weights;
            }
        }
    }

    LogResults(current_weights, truth_table);
    printf("\n Final network weights:\n");
    OutputWeights(current_weights);
    std::cout<<"(program end)\n";
}