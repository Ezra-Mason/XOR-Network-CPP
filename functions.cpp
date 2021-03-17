#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

//
//Randomly assigns the given weights as numbers between -0.5 and 0.5
//
std::vector<double> SetUpWeights(std::vector<double> weights){
    srand(time(0));
    for (int i=0; i<weights.size(); i++){
        weights[i] = (float) rand()/RAND_MAX -0.5;
    }
    return weights;
}

//
//Print the weights to the terminal screen
//
void OutputWeights(std::vector<double> weights){
    for (int i=0; i<weights.size(); i++){
        std::cout<<"w["<<i<<"] = "<<weights[i]<<"\n";
    }
    std::cout<<"(end of list)\n";
}

//
//sigmoid threshold function, bounds the node output between 0 and 1
// with inbetween values allowed  around input=0
//
double Sigmoid(double input){
    double k = 2.0; //steepness of the slope of the function
    double sigmoid = 1.0/(1.0 + exp(-k*input));
    return sigmoid;
}

//
//Determine the deviation form the given truth table for the given network weights
//
double GlobalError(std::vector<double> weights, int truth_table[8][4]){
    std::vector<double> E(8);
    double global_error = 0;
    // Attempt to reproduce the truth table with the given weights
    for(int i =0; i<E.size(); i++){
        // hidden layer outputs
        double H4 = Sigmoid(weights[0]+weights[4]*truth_table[i][0]+weights[5]*truth_table[i][1]+weights[6]*truth_table[i][2]);
        double H5 = Sigmoid(weights[1]+weights[7]*truth_table[i][0]+weights[8]*truth_table[i][1]+weights[9]*truth_table[i][2]);
        double H6 = Sigmoid(weights[2]+weights[10]*truth_table[i][0]+weights[11]*truth_table[i][1]+weights[12]*truth_table[i][2]);
        //calculate final output
        double O7 = Sigmoid(weights[3] + weights[13]*H4 + weights[14]*H5 + weights[15]*H6);
        //Error from difference between value and truth table
        E[i] = (truth_table[i][3]-O7)*(truth_table[i][3]-O7);
        // std::cout<<"E["<<i<<"]"<<E[i]<<std::endl;
        //sum the individual error for each row of the truth table
        global_error = global_error + E[i];
        
    }
    global_error = global_error * 0.5;

    return global_error;
}

//
// Log the final results of the network against the truth table
//
void LogResults(std::vector<double> weights, int truth_table[8][4]){
    printf("Truth table and final network results: \n I1, I2, I3, XOR Out, Net. O7 \n");
    for (unsigned short int i = 0; i < 8; i++)
    {
        double H4 = Sigmoid(weights[0]+weights[4]*truth_table[i][0]+weights[5]*truth_table[i][1]+weights[6]*truth_table[i][2]);
        double H5 = Sigmoid(weights[1]+weights[7]*truth_table[i][0]+weights[8]*truth_table[i][1]+weights[9]*truth_table[i][2]);
        double H6 = Sigmoid(weights[2]+weights[10]*truth_table[i][0]+weights[11]*truth_table[i][1]+weights[12]*truth_table[i][2]);
        //calculate final output
        double O7 = Sigmoid(weights[3] + weights[13]*H4 + weights[14]*H5 + weights[15]*H6);
        // print out results
         printf("%2i, %2i, %2i, %5i,     %4.2f \n",
         truth_table[i][0], truth_table[i][1], truth_table[i][2], truth_table[i][3],round(O7));

    }
    std::cout<<"(end log)"<<std::endl;

}
