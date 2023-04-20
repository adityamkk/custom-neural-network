'use strict';

/*
    GLOBAL VARIABLES, OBJECTS, AND FUNCTIONS
*/


/*
* Description: object containing activation function containers, each of which contain their respective "normal" function and "derivative" function
*/
const activationFunctions = {
    none: {
        fn: function (num) {
            return num;
        },
        derivative: function (num) {
            return 1;
        }
    },
    relu: {
        fn: function (num) {
            if (num <= 0) { return 0; }
            else { return num; }
        },
        derivative: function (num) {
            if (num <= 0) { return 0; }
            else { return 1; }
        }
    },
    sigmoid: {
        fn: function (num) {
            return (1 / (1 + Math.pow(Math.E, (-1) * num)));
        },
        derivative: function (num) {
            return (Math.pow(Math.E, (-1) * num) / Math.pow(1 + Math.pow(Math.E, (-1) * num), 2));
        }
    }
}

/* 
* Description: generates an array of randomized weights of a particular size
* Precondition: "size" is an integer greater than 0
* Postcondition: an array of randomly generated floating-point values ranging from [-4, 4) of a length of "size"
*/
function generateRandomWeights(size) {
    let arr = [];
    for (let i = 0; i < size; i++) {
        arr.push(8 * Math.random() - 4);
    }
    return arr;
}

/*
* Description: generates a random bias
* Postcondition: generates a random floating-point value ranging from [-1, 1)
*/
function generateRandomBias() {
    return 2 * Math.random() - 1;
}

/*
* Description: finds all negative partial derivatives of the cost function with respect to the output layer
* Precondition: "predicted" and "actual" are both arrays of floating-point values with lengths greater than zero, and are both of the same length
* Postcondition: returns an array of the calculated negative partial derivatives of the cost function with respect to each output neuron
*/
function costPartialDerivatives(predicted, actual) {
    let pdArr = [];
    for (let i = 0; i < predicted.length; i++) {
        pdArr.push(-2 * (predicted[i] - actual[i]));
    }
    return pdArr;
}

/*
    CLASS DEFINITIONS
*/

/*
* Description: Abstract class of a Neuron, serves no purpose other than polymorphism and intuitiveness
*/
class Neuron { constructor() { } }

/*
* Description: A single node in a neural network. Recieve inputs and produce an output (activation)
*/
class HiddenNeuron extends Neuron {
    /*
    * Description: Constructs a HiddenNeuron
    * Precondition: "previousLayer" is a Layer object
    *               "weights" is an array of floating-point values the same size as the number of neurons in "previousLayer"
    *               "bias" is a single floating-point number
    *               "activation" is a reference to a particular activationFunction sub-object
    * Postcondition: Constructs a HiddenNeuron with these characteristics
    */
    constructor(previousLayer, weights, bias, activation) {
        super();
        this.previousLayer = previousLayer;
        this.weights = weights;
        this.bias = bias;
        this.activation = activation;
        this.calculatedValue = 0;
        this.calculatedZeta = 0;

        this.weightChanges = [];
        for (let i = 0; i < weights.length; i++) {
            this.weightChanges.push(0);
        }
        this.biasChange = 0;
    }

    // GETTERS
    getWeights() { return this.weights; }
    getBias() { return this.bias; }
    getActivation() { return this.activation; }
    getCalculatedValue() { return this.calculatedValue; }
    getCalculatedZeta() { return this.calculatedZeta; }

    // HELPER METHODS

    /*
    * Description: Calculates the value of a neuron recursively and stores it, as well as the zeta value of the neuron. Returns the calculated value
    * Precondition: "this.weights" is not null, and the values within it are all initialized. Additionally, the reference to the previous layer exists
    * Postcondition: Stores the zeta value and the activation value within the neuron, and returns the activation value
    */
    calculateValue() {
        this.calculatedZeta = 0;
        for (let i = 0; i < this.previousLayer.getNeurons().length; i++) {
            // Find the weighted sum of the activations of the previous layer...
            this.calculatedZeta += this.weights[i] * this.previousLayer.getNeurons()[i].calculateValue();
        }
        // ... plus a bias
        this.calculatedZeta += this.bias;
        this.calculatedValue = this.getActivation().fn(this.calculatedZeta);
        return this.calculatedValue;
    }
}

/*
* Description: A single input node in a neural network. Recieve an input from the network itself.
*/
class InputNeuron extends Neuron {
    /*
    * Description: constructs an input neuron
    * Precondition: activationFunction is a sub-object of the activationFunctions object
    * Postcondition: creates an input neuron
    */
    constructor(activationFunction) {
        super();
        this.input = 0;
        this.activationFunction = activationFunction;
    }

    /*
    * Description: sets the value of the neuron by firing it through an activation function
    * Precondition: "input" is a floating-point number
    * Postcondition: the value of "this.input" is set to the corresponding "input"
    */
    setValue(input) {
        this.input = this.activationFunction.fn(input);
    }

    /*
    * Description: returns the value of "this.input"
    * Precondition: the value of "this.input" has already been set
    * Postcondition: returns the value of "this.input"
    */
    calculateValue() {
        //console.log("Input: " + this.input);
        return this.input;
    }

    /*
    * Description: returns the value of "this.input"
    * Precondition: the value of "this.input" has already been set
    * Postcondition: returns the value of "this.input"
    */
    getCalculatedValue() {
        return this.input;
    }
}

/*
* Description: a collection of neurons that make up a layer in a network
*/
class Layer {
    /*
    * Description: creates a layer of neurons of a particular size and type
    * Precondition: "previousLayer" is a reference to another Layer object, or null if the layer is an input layer
    *               "type" is a string, which is the type of neurons the layer contains (input, hidden, etc.)
    *               "numberOfNeurons" is a positive integer that describes the number of neurons in the layer
    *               "learningRate" is the learningRate of the backPropagation algorithm
    *               "activationFunction" is a sub-object of the activationFunctions object
    * Postcondition: constructs a layer of neurons with certain characteristics
    */
    constructor(previousLayer, type, numberOfNeurons, learningRate, activationFunction) {
        this.learningRate = learningRate;
        this.previousLayer = previousLayer;
        this.type = type;
        this.size = numberOfNeurons;
        this.activationFunction = activationFunction;
        if (type === "hidden") {
            this.neurons = [];
            for (let i = 0; i < numberOfNeurons; i++) {
                this.neurons.push(new HiddenNeuron(previousLayer, generateRandomWeights(previousLayer.getSize()), generateRandomBias(), activationFunction));
            }
        } else if (type === "input") {
            this.neurons = [];
            for (let i = 0; i < numberOfNeurons; i++) {
                this.neurons.push(new InputNeuron(activationFunction));
            }
        } else {
            // Dummy "else"
            this.neurons = [];
            for (let i = 0; i < numberOfNeurons; i++) {
                this.neurons.push(new HiddenNeuron(previousLayer, generateRandomWeights(previousLayer.getSize()), generateRandomBias(), activationFunction));
            }
        }
    }

    // GETTERS
    getPreviousLayer() { return this.previousLayer; }
    getNeurons() { return this.neurons; }
    getSize() { return this.size; }
    getActivationFunction() { return this.activationFunction; }
    getLearningRate() { return this.learningRate; }

    // HELPER METHODS

    /*
    * Description: calculates gradient descent for all neurons in a layer, and passes the derivatives of the activations of the previous layer backwards to the previous layer
    * Precondition: "partialDerivatives" is defined, "this.learningRate" is not zero
    * Postcondition: performs gradient descent on all weights and biases in the current layer, and calls the backPropagation algorithm of the previous layer
    */
    backPropagate(partialDerivatives) {
        // Base case: the layer is not a hidden layer (e.g. input layer)
        if (this.type === "hidden") {
            for (let i = 0; i < this.getNeurons().length; i++) {
                // Change weights
                for (let j = 0; j < this.getNeurons()[i].weights.length; j++) {
                    this.getNeurons()[i].weightChanges[j] += (this.getLearningRate()) * (this.getPreviousLayer().getNeurons()[j].getCalculatedValue()) * (this.getActivationFunction().derivative(this.getNeurons()[i].getCalculatedZeta())) * (partialDerivatives[i]);
                }
                // Change bias
                this.getNeurons()[i].biasChange += (this.getLearningRate()) * (this.getActivationFunction().derivative(this.getNeurons()[i].getCalculatedZeta())) * (partialDerivatives[i]);
            }
            // For each neuron in the previous layer, add the current layer's partial derivatives for the neuron. Store in an array.
            const pDNew = [];
            for (let i = 0; i < this.getPreviousLayer().getNeurons().length; i++) {
                let pDCurrent = 0;
                for (let j = 0; j < this.getNeurons().length; j++) {
                    pDCurrent += (this.getLearningRate()) * (this.getNeurons()[j].getWeights()[i]) * (this.getActivationFunction().derivative(this.getNeurons()[j].getCalculatedZeta())) * (partialDerivatives[j]);
                }
                pDNew.push(pDCurrent);
            }
            // Call the backPropagation algorithm for the previous layer
            this.getPreviousLayer().backPropagate(pDNew);
        }
    }

    /*
    * Description: applies the changes described in each neuron's change variables to each weight and bias for each neuron in the layer
    * Precondition: "dataSize" is a positive integer
    * Postcondition: adjusts each weight and bias within a layer accordingly
    */
    adjustWeights(dataSize) {
        for (let i = 0; i < this.getNeurons().length; i++) {
            for (let j = 0; j < this.getNeurons()[i].getWeights().length; j++) {
                this.getNeurons()[i].weights[j] += this.getNeurons()[i].weightChanges[j] / dataSize;
                this.getNeurons()[i].weightChanges[j] = 0;
            }
        }
    }
}

/*
* Description: Main architectural framework for storing all the layers of the network in place, and runs the methods for training and testing the model
*/
class NeuralNetwork {
    /*
    * Description: initializes a new neural network
    * Precondition: "iterations" is a positive integer
    *               "learningRate" is a floating-point value in the range [0, 1)
    *               "layers" is an array of objects with three instance variables: type, size, and activation
    *                   "type" is a string which indicates the type of layer it is
    *                   "size" is a positive integer indicating the number of neurons in the layer
    *                   "activation" is a reference to a sub-object of the activationFunctions object
    * Postcondition: initializes a new neural network
    */
    constructor(iterations, learningRate, layers) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.layers = [];
        let prevLayer = null;
        for (let i = 0; i < layers.length; i++) {
            let layer;
            if (layers[i].type === "input") {
                layer = new Layer(prevLayer, "input", layers[i].size, learningRate, layers[i].activation);
                prevLayer = layer;
            } else {
                layer = new Layer(prevLayer, "hidden", layers[i].size, learningRate, layers[i].activation);
                prevLayer = layer;
            }
            this.layers.push(layer);
        }
    }

    // GETTERS
    getLayers() { return this.layers; }
    getIterations() { return this.iterations; }
    getLearningRate() { return this.learningRate; }

    //HELPER METHODS

    /*
    * Description: performs iterations through data, performing backPropagation and merging all backPropagation for each iteration
    * Precondition: "data" is an array of objects which each contain an "input" array and "output" array
    * Postcondition: performs iterations through data, performing backPropagation and merging all backPropagation for each iteration
    *                prints the iteration number and the average value of the cost function for each iteration
    */
    train(data) {
        for (let iter = 0; iter < this.getIterations(); iter++) {
            let iterCost = 0;
            for (let i = 0; i < data.length; i++) {
                // Make prediction on input data
                const predicted = this.predict(data[i].input);
                iterCost += this.cost(predicted, data[i].output) / data.length;
                const layer = this.getLayers()[this.getLayers().length - 1];
                // Perform backPropagation with respect to the predicted output values
                layer.backPropagate(costPartialDerivatives(predicted, data[i].output));
            }
            // Print average value of cost function to the console
            console.log(`Iteration ${iter + 1}: COST = ${iterCost}`);
            // Adjust weights after one iteration
            for (let i = 1; i < this.getLayers().length; i++) {
                this.getLayers()[i].adjustWeights(data.length);
            }
        }
    }

    /*
    * Description: returns the values of the output neurons after feeding in input
    * Precondition: "inputs" is an array of floating-point values that is the same size as the number of neurons in the input layer
    * Postcondition: returns an array of predicted values of the neural network, given its current weights and biases
    */
    predict(inputs) {
        // Sets the values of the input layer
        for (let i = 0; i < this.getLayers()[0].getNeurons().length; i++) {
            this.getLayers()[0].getNeurons()[i].setValue(inputs[i]);
        }

        const predicted = [];
        // Calculates the predicted values in the output layer
        for (let i = 0; i < this.getLayers()[this.getLayers().length - 1].getNeurons().length; i++) {
            let pred = this.getLayers()[this.getLayers().length - 1].getNeurons()[i].calculateValue();
            predicted.push(pred);
        }

        return predicted;
    }

    /*
    * Description: calculates the value of the cost function between predicted and actual values
    * Precondition: "predicted" and "actual" are both arrays of floating-point numbers of the same size and both have a length greater than 0
    * Postcondition: returns the value of the cost function
    */
    cost(predicted, actual) {
        let cost = 0;
        for (let i = 0; i < predicted.length; i++) {
            cost += Math.pow(predicted[i] - actual[i], 2);
        }
        return cost;
    }
}

/*
    EXECUTABLE CODE
*/

let myNN = null;

/*
// Inverter
myNN.train([
    { input: [1, 1, 1], output: [0, 0, 0] },
    { input: [1, 1, 0], output: [0, 0, 1] },
    { input: [1, 0, 1], output: [0, 1, 0] },
    { input: [1, 0, 0], output: [0, 1, 1] },
    { input: [0, 1, 1], output: [1, 0, 0] },
    { input: [0, 1, 0], output: [1, 0, 1] },
    { input: [0, 0, 1], output: [1, 1, 0] },
    { input: [0, 0, 0], output: [1, 1, 1] },
]);

console.log(`Predicted : [${myNN.predict([1, 0, 0])}], Actual : 0, 1, 1`);
console.log(`Predicted : [${myNN.predict([0, 0, 1])}], Actual : 1, 1, 0`);
*/

/*
// Contrast
myNN.train([
    { input : [0.03, 0.7, 0.5], output : [0] },
    { input : [0.16, 0.09, 0.2], output : [1] },
    { input : [0.5, 0.5, 1.0], output : [1] }
]);

console.log(`Predicted : ${myNN.predict([1, 0.4, 0])}, Actual : 1`);
*/

/*
    DOM MANIPULATION
*/


document.getElementById("number-of-layers").addEventListener("change", (event) => {
    document.getElementById("layers").innerHTML = "";
    for (let i = 0; i < event.target.value; i++) {
        document.getElementById("layers").innerHTML += `<tr><td><label for=\"size-${i}\">Size: </label><input type=\"number\" id=\"size-${i}\" name=\"size-${i}\"></td><td><label for=\"activation-${i}\">Activation: </label><select name=\"activation-${i}\" id=\"activation-${i}\"><option value=\"none\">None</option><option value=\"relu\">Relu</option><option value=\"sigmoid\">Sigmoid</option></select></td></tr>`;
    }

    document.getElementById("predict-input").innerHTML = "";
});

document.getElementById("data-num").addEventListener("change", (event) => {
    let inputString = "";
    let outputString = ""
    for (let i = 0; i < event.target.value; i++) {
        inputString += `<tr>`;
        outputString += `<tr>`;
        for (let j = 0; j < document.getElementById("size-0").value; j++) {
            inputString += `<td>`;
            inputString += `<input type=\"number\" id=\"data-input-${i}-${j}\">`;
            inputString += `</td>`;
        }
        for (let j = 0; j < document.getElementById(`size-${document.getElementById("number-of-layers").value - 1}`).value; j++) {
            outputString += `<td>`;
            outputString += `<input type=\"number\" id=\"data-output-${i}-${j}\">`;
            outputString += `</td>`;
        }
        inputString += `</tr>`;
        outputString += `</tr>`;
    }
    document.getElementById("data-input").innerHTML = inputString;
    document.getElementById("data-output").innerHTML = outputString;

    document.getElementById("predict-input").innerHTML = "";
});

document.getElementById("train").addEventListener("click", (event) => {
    let layers = [];
    for(let i = 0; i < document.getElementById("number-of-layers").value; i++) {
        const myLayer = {type: "", size: 1, activation: activationFunctions.none};
        if(i === 0) {
            myLayer.type = "input";
        } else if(i == document.getElementById("number-of-layers").value-1) {
            myLayer.type = "output";
        } else {
            myLayer.type = "hidden";
        }
        myLayer.size = document.getElementById(`size-${i}`).value;
        if(document.getElementById(`activation-${i}`).value === "none") {
            myLayer.activation = activationFunctions.none;
        } else if(document.getElementById(`activation-${i}`).value === "relu") {
            myLayer.activation = activationFunctions.relu;
        } else if(document.getElementById(`activation-${i}`).value === "sigmoid") {
            myLayer.activation = activationFunctions.sigmoid;
        } else {
            myLayer.activation = activationFunctions.sigmoid;
        }
        layers.push(myLayer);
    }
    //myNN = new NeuralNetwork(document.getElementById("iterations").value, document.getElementById("learning-rate").value, [{ type: "input", size: 3, activation: activationFunctions.none }, { type: "hidden", size: 6, activation: activationFunctions.sigmoid }, { type: "hidden", size: 3, activation: activationFunctions.sigmoid }]);
    myNN = new NeuralNetwork(document.getElementById("iterations").value, document.getElementById("learning-rate").value, layers);

    
    if(document.getElementById("size-0") != null) {
        let predString = "<tr>";
        for(let i = 0; i < document.getElementById("size-0").value; i++) {
            predString += `<td><input type=\"number\" id=\"pred-${i}\"></td>`;
        }
        predString += "</tr>";
        document.getElementById("predict-input").innerHTML = predString;
    }
});