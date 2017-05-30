
// NEURAL NETWORK FOR THE CELLULASE EXPERIMENT: TRIGGERING THE STEP UP OF THE C/N FEEDING BASED ON OBSERVING THE PH, DO & ORP INPUTS

/* The system has three basic inputs (namely pH, DO & ORP), from which four more extra inputs are derived, summing up seven
inputs in total. The system is trained by observing stepping up decisions made by human judgment during the experiments. For
modelling the triggering action, the output can take values from 0 to 1, with values above a threshold triggering the step up
and higher confidence the higher the value is. The threshold value will be chosen according to the training results, as the
value that provides a bigger error margin to the system will be the best one, since its robustness will be the biggest. */

// LIBRARIES included in the program //

#include <iostream> // For screen outputs
#include <sstream>  // Helps to the automation of the file names
#include <fstream>  // Reading and writing from/to files
#include <vector>   // The data is structured in vectors
#include <stdlib.h> // For random number generation
#include <time.h>   // Seed for random numbers
#include <math.h>   // Math operations


//DEFINE commands set the parameters for the program // ***Read all of them carefully before starting the program***

// Program parameters
#define FLAG_TRAIN_TEST     1       // '0' = perform training, '1' = test performance of previously trained system
#define TRAIN_FILE          29      // Label for the current training instance. For generating files with training data
                                    // or choosing a system for loading its weights for testing
// Network architecture
#define INPUT_NUMBER        7       // Number of inputs of the system
#define	HIDDEN_NUMBER	    8       // Number of neurons at the hidden layer of the neural network

// Training hyper-parameters
#define LEARN_COEFFICIENT   0.7     // Learning rate coefficient (training speed)
#define WEIGHT_DECAY        0.00005 // Regularization parameter (improves generalization)
#define SLOPE               1       // Slope of sigmoid function. Default is 1.
#define INITIAL_WEIGHT_RANGE 0.87   // Range of initial weight values. Default is 1
#define MAX_TRAIN_TIMES     100000  // Number of epochs of the training


/* FILE DETAILS

"input_EXP'X'.txt" file must have X+1 columns, where X is the number of inputs variables to the system. The last
column corresponds to the desired output (y) that acts as a label for supervised training.

The number at the rest of the files are generated for labelling purposes (TRAIN_FILE). 'test_output'XXX'.txt' has a first digit
corresponding to the number of the EXP in which the test was performed (input_EXP'X'), the others are the label
'XX' of the training set employed.

*/

using namespace std;

typedef struct{     //**Scaling parameters**
    double offset;  // Constant value substracted to the input. Around MEAN_INPUT_VALUE is best
    double factor;  // Constant by which the input is divided. Around 1/MAX_INPUT_VALUE is best
    double max;     // Maximum value that the input can reach (saturation)
} SCALING;

typedef struct{     //**Names of the files used for i/o data**
    string input, train_output, mse, results, weights, test_output;
} FILES;

typedef struct{     //**System parameters**
    vector<int> train_experiments, test_experiments;    // Vector for experiments used for training, same for testing
    SCALING  scaling[INPUT_NUMBER+1]; // Scaling parameters for all the inputs and the output
    FILES files;                      // Names for all files in the program
    int train_file, hidden_number;    // TRAIN_FILE, HIDDEN_NUMBER. Allows variations in this parameters for grid search
    double weight_decay;              // WEIGHT_DECAY. Allows variations in this parameters for grid search
} PARAM;

typedef vector<double> VEC; // Each vector corresponds to a training sample and has X+1 components: X values corresponding
                            // to the different inputs and one for the output
typedef vector<VEC> MAT;    // The matrix contains all the training samples

void Initialize_File_Names                                                                   (PARAM* param);
void Initialize_Parameters                                                                   (PARAM* param);
void Train_System             (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[],PARAM* param);
int Load_Weights              (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[],PARAM* param);
void Cross_Test               (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[],PARAM* param,           int exp_number);

void Initial_Neural_Weight    (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[]);
void Read_Input                                                                              (PARAM* param,MAT& Input,int exp_number);
void Back_Propagation_Learning(double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[],PARAM* param,MAT& Input);
void Save_Results             (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[],PARAM* param);

double Activation_Function     (double input);
double Activation_Function_Diff(double input);

int main()
{
    double Hidden_Weight[HIDDEN_NUMBER][INPUT_NUMBER+1];
    double Output_Weight[HIDDEN_NUMBER+1];
    PARAM param;
    int flag_weights = 1;

    Initialize_Parameters                                 (&param);

    //for(int i=0; i<3; i++){ //For grid search of hyper-parameters. i<1 for regular training

        Initialize_File_Names                             (&param);

        if(!FLAG_TRAIN_TEST){

            Train_System    (Hidden_Weight, Output_Weight, &param); //Train system

        }else{

            flag_weights = Load_Weights    (Hidden_Weight, Output_Weight, &param); //Load weights form previous trained system

        }

        if(flag_weights){ // Test is only performed if there were no errors at loading weights

            for(int i=0; i<param.test_experiments.size(); i++){ //Perform test on test experiments
                Cross_Test      (Hidden_Weight, Output_Weight, &param, param.test_experiments[i]);
            }
        }

        param.train_file++;
        param.hidden_number++;
        //param.weight_decay/=2;

    //}

    return 0;
}

void Initialize_File_Names    (PARAM* param)
{
    stringstream ss1, ss2, ss3, ss4, ss5, ss6;

    ss1<<"input_EXP";
	ss1>>param->files.input;
	//cout<<param->files.input<<"\n";           // Debug purposes

	ss2<<"mse"<<param->train_file<<".txt";
	ss2>>param->files.mse;
	//cout<<param->files.mse<<endl;             // Debug purposes

	ss3<<"train_output"<<param->train_file<<".txt";
	ss3>>param->files.train_output;
    //cout<<param->files.train_output<<endl;    // Debug purposes

	ss4<<"results"<<param->train_file<<".txt";
	ss4>>param->files.results;
	//cout<<param->files.results<<endl;         // Debug purposes

    ss5<<"weights"<<param->train_file<<".txt";
	ss5>>param->files.weights;
	//cout<<param->files.weights<<endl;         // Debug purposes

    ss6<<"test_output";
	ss6>>param->files.test_output;
	//cout<<param->files.test_output<<endl;     // Debug purposes
}

void Initialize_Parameters    (PARAM* param)
{

    param->train_experiments.push_back(3);  // Select which input files (iput_EXPX.txt) will be used for training
    param->train_experiments.push_back(5);
    param->train_experiments.push_back(6);
    param->train_experiments.push_back(8);
    //param->train_experiments.push_back(8);
    //param->train_experiments.push_back(8);
    //param->train_experiments.push_back(8);
    //param->train_experiments.push_back(8);
    //param->train_experiments.push_back(8);
    //param->train_experiments.push_back(8);
    param->train_experiments.push_back(9);

    cout<<"\nTraining experiments: ";
    for(int i=0; i<param->train_experiments.size(); i++){ // Debugging purposes
        cout<< param->train_experiments[i]<<", ";
    }

    param->test_experiments.push_back(1);
    param->test_experiments.push_back(2);   // Select which input files (iput_EXPX.txt) will be used for testing
    param->test_experiments.push_back(4);
    param->test_experiments.push_back(7);
    //param->test_experiments.push_back(9);

    cout<<"\nTest experiments: ";
    for(int i=0; i<param->test_experiments.size(); i++){  // Debugging purposes
        cout<< param->test_experiments[i]<<", ";
    }

    param->scaling[0].offset= 5.5;  // pH
    param->scaling[0].factor= 0.1;
    param->scaling[0].max  = 5.6;

    param->scaling[1].offset= 0.0075;// Smoothed diff pH
    param->scaling[1].factor= 0.0075;
    param->scaling[1].max  = 0.03;

    param->scaling[2].offset= 25;   // DO
    param->scaling[2].factor= 25;
    param->scaling[2].max  = 70;

    param->scaling[3].offset= 15;   // Smoothed diff DO
    param->scaling[3].factor= 15;
    param->scaling[3].max  = 45;

    param->scaling[4].offset= 150;  // ORP
    param->scaling[4].factor= 150;
    param->scaling[4].max  = 400;

    param->scaling[5].offset= 15;   // Diff ORP
    param->scaling[5].factor= 25;
    param->scaling[5].max  = 40;

    param->scaling[6].offset= 0.5;  // Smoothed diff ORP
    param->scaling[6].factor= 1.5;
    param->scaling[6].max= 2;

    param->scaling[7].offset= 0.5;  // Target output
    param->scaling[7].factor= 0.5;
    param->scaling[7].max= 1;

    param->hidden_number = HIDDEN_NUMBER;
    cout<<"\nNumber of neurons at the hidden layer: "<<param->hidden_number;

    param->train_file = TRAIN_FILE;
    cout<<"\nTrain file number: "<<param->train_file;

    param->weight_decay = WEIGHT_DECAY;
    cout<<"\nRegularization parameter: "<<param->weight_decay;

    for(int i=0; i<INPUT_NUMBER+1;i++){   // Debugging purposes
        cout<<endl<<i<<"th variable scaling: offset "<< param->scaling[i].offset <<", factor "<< param->scaling[i].factor
                     <<", maximum "<< param->scaling[i].max;
    }

}

void Train_System             (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[], PARAM* param)
{
	MAT Input;

    Initial_Neural_Weight     (Hidden_Weight,Output_Weight); //Random initialization of weights in the range of INITIAL WEIHGTS RANGE

    for(int i=0; i<param->train_experiments.size(); i++){   // All the experiments used for training are stored in MAT Input
        Read_Input            (                            param,Input,param->train_experiments[i]);
    }

    Back_Propagation_Learning (Hidden_Weight,Output_Weight,param,Input);    // Train with backprop
	Save_Results              (Hidden_Weight,Output_Weight,param      );    // Write 'results' file and 'weights' file
}

//return 0: reading successful, return 1: failure reading
int Load_Weights              (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[], PARAM* param)
{
    int input_n, hidden_n; // Number of inputs and number of neurons in hidden layer, respectively

    fstream weight_file;
    weight_file.open(param->files.weights.c_str()); // Opens weight file with the values to be loaded

    weight_file>>input_n>>hidden_n;     // Reads the first two values of file, indicating input number and number of hidden neurons
    cout<<"\ninput number "<<input_n<<" = "<<INPUT_NUMBER<<",  hidden neurons number "<<hidden_n<<" = "<<HIDDEN_NUMBER<<"\n";

    for(int i=0; i<INPUT_NUMBER;i++){   // Reads scaling information of the weights being loaded
        weight_file >> param->scaling[i].offset >> param->scaling[i].factor >> param->scaling[i].max;
        cout<<i<<" Variable Scaling: offset "<<param->scaling[i].offset<<", factor "<<param->scaling[i].factor<<", maximum "<<param->scaling[i].max<<"\n";
    }

    if((input_n == INPUT_NUMBER) && (hidden_n == param->hidden_number)){ // Checks if weights configuration coincides with current test configuration
        // Weights and bias loading
        for(int j=0; j<param->hidden_number; j++){
                weight_file>>Hidden_Weight[j][0];
                cout<<"\n"<< Hidden_Weight[j][0];

            for(int k=0;k<INPUT_NUMBER;k++){
                weight_file>>Hidden_Weight[j][k+1];
                cout<<" "<<  Hidden_Weight[j][k+1];
            }
        }
        weight_file>>        Output_Weight[0];
        cout<<"\n"<<         Output_Weight[0];
        for(int j=0;j<param->hidden_number;j++){
            weight_file>>    Output_Weight[j+1];
            cout<<" "<<      Output_Weight[j+1];
        }

    }else{ // If configurations are not same display error message

        cout<<"\nError: input number "<<input_n<<" != "<<INPUT_NUMBER<<"  OR  "
        "hidden neurons number "<<hidden_n<<" != "<<param->hidden_number<<"\n";

        return 0;
    }

    return 1;
}

void Cross_Test               (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[], PARAM* param, int exp_number)
{
	double z1[param->hidden_number], z2, a1[param->hidden_number], a2; // Inner values of network
    stringstream ss;
    string input_file_name;
    MAT Input;

    Read_Input( param, Input, exp_number); // Read data from EXP'exp_number'

    ss<<param->files.test_output<<exp_number<<param->train_file<<".txt"; // First digit is experiment number, next, training label
    ss>>input_file_name;
    fstream test_output(input_file_name.c_str(),fstream::out); // Create output file

    // Feedforward
	for(int i=0; i<Input.size(); i++)
    {
        z2 = Output_Weight[0]; //Bias of the output
        for(int j=0; j<param->hidden_number; j++)
        {
            z1[j]= Hidden_Weight[j][0]; //Bias of the hidden neuron j
            for(int k=0;k<INPUT_NUMBER;k++)
                z1[j] += Hidden_Weight[j][k+1] * Input[i][k]; //z1(j)= weight(jk)· input(i)(k)
            a1[j] = Activation_Function(z1[j]);             // a1(j)= f(z1(j))
            z2 += Output_Weight[j+1]* a1[j];                // z2 = weight(j)· output(j)
        }
        a2 = Activation_Function(z2);                       // a2 = f(z2)  (System Output)
        test_output<<a2<<" ";
	}

}

void Initial_Neural_Weight    (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[])
{
	srand( (unsigned)time( NULL ) ); // Seed of random number generation

	for(int j=0; j<HIDDEN_NUMBER; j++){
        for(int k=0; k<INPUT_NUMBER+1; k++)
            Hidden_Weight[j][k] =   ((double)rand()/RAND_MAX-0.5) * 2 * INITIAL_WEIGHT_RANGE ; // [-r,+r]
	}
	for(int j=0;j<HIDDEN_NUMBER;j++)
            Output_Weight[j] =      ((double)rand()/RAND_MAX-0.5) * 2 * INITIAL_WEIGHT_RANGE ;

////Debugging code
//
//	for(int j=0; j<HIDDEN_NUMBER; j++){
//
//        cout<<"\n"<< Hidden_Weight[j][0];
//        for(int k=0;k<INPUT_NUMBER;k++)
//            cout<<" "<<  Hidden_Weight[j][k+1];
//
//    }
//
//    cout<<"\n"<<         Output_Weight[0];
//    for(int j=0;j<HIDDEN_NUMBER;j++)
//        cout<<" "<<      Output_Weight[j+1];
}

void Read_Input               (PARAM* param, MAT& Input, int exp_number)
{
    stringstream ss;
    string input_file_name;
    VEC file_reader(INPUT_NUMBER+1);            // Auxiliar variable for reading values from the input file

    ss<<param->files.input<<exp_number<<".txt";
    ss>>input_file_name;
    fstream input_file(input_file_name.c_str());// Open input file
    cout<<endl<<input_file_name;

    while(input_file>>file_reader[0]) // If there are still values to be read
    {
        if(file_reader[0] > param->scaling[0].max)  // Scaling
            file_reader[0] = param->scaling[0].max;
        file_reader[0]   -= param->scaling[0].offset;
        file_reader[0]   /= param->scaling[0].factor;
        //cout<<setw(10);
        //cout<<file_reader[0]<<" ";

        for(int i=1; i<INPUT_NUMBER+1;i++){ //INPUT_NUMBER+1 (inputs + output) data per line
            input_file >> file_reader[i];
            if(file_reader[i] > param->scaling[i].max) // Scaling
               file_reader[i] = param->scaling[i].max;
            file_reader[i]   -= param->scaling[i].offset;
            file_reader[i]   /= param->scaling[i].factor;
            //cout<<setw(10);
            //cout<<file_reader[i]<<" "; //Debugging
        }

        Input.push_back(file_reader); // Add values read to MAT Input
    }
    cout<<endl<<"Input SIze: "<<Input.size()<<endl<<endl;
}

void Back_Propagation_Learning(double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[], PARAM* param, MAT& Input)
{
    double z1[param->hidden_number],z2,a1[param->hidden_number],a2,error,e1[param->hidden_number],delta1[param->hidden_number],e2,delta2,Average_Delta;
	double H_Grad[param->hidden_number][INPUT_NUMBER+1]; // param->hidden_number neurons, each of them with INPUT_NUMBER+1 weights (inputs+bias)
    double O_Grad[param->hidden_number+1];  // param->hidden_number+1 weights of the output neuron (hidden nerurons + bias)
    int counter=0;                          // Number of epochs

    ofstream mse(param->files.mse.c_str(),fstream::out);
    ofstream train_output(param->files.train_output.c_str(),fstream::out); // Create output file

    cout<<"\nInput size: "<<Input.size()<<endl;  // Total number of training samples

    while(counter<MAX_TRAIN_TIMES)
	{
        error = 0;
        counter++;

        // Initialize gradients
                O_Grad[0]       =0;         //Output bias gradient = 0
        for(int j=0;j<param->hidden_number;j++)
        {
                O_Grad[j+1]     =0;         //Output weight gradients = 0
                H_Grad[j][0]    =0;         //Hidden bias gradient = 0
            for(int k=0;k<INPUT_NUMBER;k++)
                H_Grad[j][k+1]  =0;         //Hidden weight gradients = 0
        }

        for(int i=0; i<Input.size(); i++)
        {
            //Feedforward
            z2 = Output_Weight[0];          //Bias of the output
            for(int j=0; j<param->hidden_number; j++)
            {
                z1[j]= Hidden_Weight[j][0]; //Bias of the hidden neuron j
                for(int k=0;k<INPUT_NUMBER;k++)
                    z1[j] += Hidden_Weight[j][k+1] * Input[i][k]; //z1 = w·x + b; z1(j)= weight(jk)· input(i)(k) + bias(j)
                a1[j] = Activation_Function(z1[j]);             // a1(j)= f(z1(j))
                z2 += Output_Weight[j+1]* a1[j];                // z2 = w·z1 + b; z2 = weight(j)· output(j) + bias
            }
            a2 = Activation_Function(z2);                       // a2 = f(z2)

            if(counter==MAX_TRAIN_TIMES)    // Record output of the trained system
                train_output<<a2<<" ";

            //Cost function
            e2 = Input[i][INPUT_NUMBER] - a2;   // e2 = (y-a2)
            error += pow(e2,2);                 // error = (y-a2)^2/TRAINING_NUMBER

            //Back-propagation
            delta2 = e2 * Activation_Function_Diff( z2 );
            Average_Delta = 0;
            for(int j=0; j<param->hidden_number; j++)
            {
                e1[j]= Output_Weight[j+1] * delta2;
                delta1[j]= e1[j] * Activation_Function_Diff( z1[j] );
                Average_Delta += fabs( delta1[j] );
            }
            Average_Delta /= param->hidden_number;     // Threshold delta for competitive learning

            //Only accept deltas bigger than average (competitive learning)
            for(int j=0; j<param->hidden_number; j++)
            {
                if(fabs( delta1[j] ) < Average_Delta)
                    delta1[j]=0;
            }

            //Batch gradient
                O_Grad[0]           -= 2*delta2;
            for(int j=0;j<param->hidden_number;j++)
            {
                O_Grad[j+1]         -= 2*delta2    * a1[j];
                H_Grad[j][0]        -= 2*delta1[j];
                for(int k=0;k<INPUT_NUMBER;k++)
                    H_Grad[j][k+1]  -= 2*delta1[j] * Input[i][k];
            }
        }

        //Mean gradient
                O_Grad[0]       /= Input.size();
        for(int j=0; j<param->hidden_number; j++)
        {
                O_Grad[j+1]     /= Input.size();
                H_Grad[j][0]    /= Input.size();
            for(int k=0; k<INPUT_NUMBER; k++)
                H_Grad[j][k+1]  /= Input.size();
        }

        //Update the weights
            Output_Weight[0]         += LEARN_COEFFICIENT*(-1) * O_Grad[0];
        for(int j=0;j<param->hidden_number;j++){
            Output_Weight[j+1]       += LEARN_COEFFICIENT*(-1) * O_Grad[j+1];
            Hidden_Weight[j][0]      += LEARN_COEFFICIENT*(-1) * H_Grad[j][0];
            for(int k=0;k<INPUT_NUMBER;k++)
                Hidden_Weight[j][k+1]+= LEARN_COEFFICIENT*(-1) * (param->weight_decay * Hidden_Weight[j][k+1] + H_Grad[j][k+1]); //Weight decay
        }

        error /= Input.size();          // error = (y-a2)^2/TRAIN_NUMBER

        if(counter%200==0){     // Save mse value every 200 epochs
            mse<<error<<" ";
            cout<<"counter="<<counter<<"    error="<<error<<"  \n\n";
        }
	}
}

void Save_Results             (double Hidden_Weight[][INPUT_NUMBER+1], double Output_Weight[], PARAM* param)
{
	fstream results(param->files.results.c_str(),fstream::out); // Open results file (for writing the training parameters)
	fstream weights(param->files.weights.c_str(),fstream::out); // Open weights file (for writing value of weights and bias)

	//Record and display system parameters
	results<<"System parameters"
         "\nInput number: "         <<INPUT_NUMBER<<
         "\nHidden number: "        <<param->hidden_number<<
         "\nLearning coefficient: " <<LEARN_COEFFICIENT<<
         "\nRegularization parameter: "<<param->weight_decay<<
         "\nSlope: "                <<SLOPE<<
         "\nInitial Weight: "       <<INITIAL_WEIGHT_RANGE<<
         "\nMax train number: "     <<MAX_TRAIN_TIMES;

    cout<<"System parameters"
         "\nInput number: "         <<INPUT_NUMBER<<
         "\nHidden number: "        <<param->hidden_number<<
         "\nLearning coefficient: " <<LEARN_COEFFICIENT<<
         "\nRegularization parameter: "<<param->weight_decay<<
         "\nSlope: "                <<SLOPE<<
         "\nInitial Weight: "       <<INITIAL_WEIGHT_RANGE<<
         "\nMax train number: "     <<MAX_TRAIN_TIMES;

    for(int i=0; i<INPUT_NUMBER;i++){
        results<<"\n"<<i<<" Variable Scaling: offset "<<param->scaling[i].offset<<", factor "<<param->scaling[i].factor<<", maximum "<<param->scaling[i].max;
        cout<<"\n"<<i<<" Variable Scaling: offset "<<param->scaling[i].offset<<", factor "<<param->scaling[i].factor<<", maximum "<<param->scaling[i].max;
    }

    //Record and display weight values
    weights<<INPUT_NUMBER<<" "<<param->hidden_number;
    for(int i=0; i<INPUT_NUMBER;i++)
        weights << "\n" << param->scaling[i].offset << " " << param->scaling[i].factor << " " << param->scaling[i].max;

	results<<"\n\nWeights of the hidden layer:";
	for(int j=0;j<param->hidden_number;j++)
	{
	    results<<"\nNeuron"<<j<<": Bias= "  <<Hidden_Weight[j][0]<<", ";
	    weights<<"\n"                       <<Hidden_Weight[j][0];

	    for(int k=0;k<INPUT_NUMBER;k++){
            results<<"Weight"<<k<<"= "      <<Hidden_Weight[j][k+1]<<", ";
            weights<<" "                    <<Hidden_Weight[j][k+1];
	    }
	}
    results<<"\nWeights of the output layer:\nBias= "<< Output_Weight[0]<<", Weights= ";
    weights<<"\n"<<                                     Output_Weight[0];
	for(int j=0;j<param->hidden_number;j++){
        results<<                                       Output_Weight[j+1]<<", ";
		weights<<" "<<                                  Output_Weight[j+1];
	}
}

double Activation_Function    (double input)
{
	double output;

    output=2/(1+exp((-1)*SLOPE*input))-1.0; // Sigmoid activation function

    //output= 1.7159*tanh(0.666667*input); // Hyperbolic function

	return output;
}

double Activation_Function_Diff(double input)
{
	double output;

	output=2*SLOPE*exp((-1)*SLOPE*input)*pow(1+exp((-1)*SLOPE*input),-2); // Sigmoid function

    //output= 1.7159*0.666667*(1/pow(cosh(0.666667*input),2)); // Hyperbolic tangent

	return output;
}
