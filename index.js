'use strict';

/*
    GLOBAL VARIABLES, OBJECTS, AND FUNCTIONS
*/

const learningRate = 1;

const activationFunctions = {
    relu : function(num) {
        if(num <= 0) {return 0;}
        else {return num;}
    },
    sigmoid : function(num) {
        return (1/(1+Math.pow(Math.E,(-1)*num)));
    },
    sigmoidDerivative : function(num) {
        return (Math.pow(Math.E,(-1)*num)/Math.pow(1+Math.pow(Math.E,(-1)*num),2));
    },
    none : function(num) {
        return num;
    }
}

function generateRandomWeights(size) {
    let arr = [];
    for(let i = 0; i < size; i++) {
        arr.push(8*Math.random()-4);
    }
    return arr;
}

function generateRandomBias() {
    return 2*Math.random()-1;
}

function costPartialDerivatives(predicted, actual) {
    let pdArr = [];
    for(let i = 0; i < predicted.length; i++) {
        pdArr.push(-2*(predicted[i] - actual[i]));
    }
    return pdArr;
}

/*
    CLASS DEFINITIONS
*/

class Neuron {constructor() {}}

class HiddenNeuron extends Neuron {
    constructor(previousLayer, weights, bias, activation) {
        super();
        this.previousLayer = previousLayer;
        this.weights = weights;
        this.bias = bias;
        this.activation = activation;
        this.calculatedValue = 0;
        this.calculatedZeta = 0;

        this.weightChanges = [];
        for(let i = 0; i < weights.length; i++) {
            this.weightChanges.push(0);
        }
        this.biasChange = 0;
    }

    // GETTERS
    getWeights() {return this.weights;}
    getBias() {return this.bias;}
    getActivation() {return this.activation;}
    getCalculatedValue() {return this.calculatedValue;}
    getCalculatedZeta() {return this.calculatedZeta;}

    // SETTERS


    // HELPER METHODS

    // Calculates the value of a neuron recursively and stores it, as well as the zeta value of the neuron. Returns the calculated value
    calculateValue() {
        this.calculatedZeta = 0;
        for(let i = 0; i < this.previousLayer.getNeurons().length; i++) {
            this.calculatedZeta += this.weights[i]*this.previousLayer.getNeurons()[i].calculateValue();
        }
        this.calculatedZeta += this.bias;
        this.calculatedValue = activationFunctions.sigmoid(this.calculatedZeta);
        //console.log("Hidden Layer: " + this.calculatedValue);
        return this.calculatedValue;
    }
}

class InputNeuron extends Neuron {
    constructor(activationFunction) {
        super();
        this.input = 0;
        this.activationFunction = activationFunction;
    }
    setValue(input) {
        this.input = this.activationFunction(input);
    }

    calculateValue() {
        //console.log("Input: " + this.input);
        return this.input;
    }

    getCalculatedValue() {
        return this.input;
    }
}

class Layer {
    constructor(previousLayer, type, numberOfNeurons, activationFunction) {
        this.previousLayer = previousLayer;
        this.type = type;
        this.size = numberOfNeurons;
        if(type === "hidden") {
            this.neurons = [];
            for(let i = 0; i < numberOfNeurons; i++) {
                this.neurons.push(new HiddenNeuron(previousLayer, generateRandomWeights(previousLayer.getSize()), generateRandomBias(), activationFunction));
            }
        } else if(type === "input"){
            this.neurons = [];
            for(let i = 0; i < numberOfNeurons; i++) {
                this.neurons.push(new InputNeuron(activationFunction));
            }
        } else {
            // Dummy "else"
            this.neurons = [];
            for(let i = 0; i < numberOfNeurons; i++) {
                this.neurons.push(new HiddenNeuron(previousLayer, generateRandomWeights(previousLayer.getSize()), generateRandomBias(), activationFunction));
            }
        }
    }

    getPreviousLayer() {return this.previousLayer;}
    getNeurons() {return this.neurons;}
    getSize() {return this.size;}

    // HELPER METHODS
    backPropogate(partialDerivatives) {
        if(this.type === "hidden") {
            for(let i = 0; i < this.getNeurons().length; i++) {
                // Change weights
                for(let j = 0; j < this.getNeurons()[i].weights.length; j++) {
                    this.getNeurons()[i].weightChanges[j] += (learningRate)*(this.getPreviousLayer().getNeurons()[j].getCalculatedValue())*(activationFunctions.sigmoidDerivative(this.getNeurons()[i].getCalculatedZeta()))*(partialDerivatives[i]);
                }
                // Change bias
                this.getNeurons()[i].biasChange += (learningRate)*(activationFunctions.sigmoidDerivative(this.getNeurons()[i].getCalculatedZeta()))*(partialDerivatives[i]);
            }
            // For each neuron in the previous layer, add the current layer's partial derivatives for the neuron. Store in an array
            const pDNew = [];
            for(let i = 0; i < this.getPreviousLayer().getNeurons().length; i++) {
                let pDCurrent = 0;
                for(let j = 0; j < this.getNeurons().length; j++) {
                    pDCurrent += (learningRate)*(this.getNeurons()[j].getWeights()[i])*(activationFunctions.sigmoidDerivative(this.getNeurons()[j].getCalculatedZeta()))*(partialDerivatives[j]);
                }
                pDNew.push(pDCurrent);
            }
            this.getPreviousLayer().backPropogate(pDNew);
        }
    }

    adjustWeights(dataSize) {
        for(let i = 0; i < this.getNeurons().length; i++) {
            for(let j = 0; j < this.getNeurons()[i].getWeights().length; j++) {
                this.getNeurons()[i].weights[j] += this.getNeurons()[i].weightChanges[j]/dataSize;
                this.getNeurons()[i].weightChanges[j] = 0;
            }
        }
    }
}

class NeuralNetwork {
    constructor(layers) {
        this.layers = [];
        let prevLayer = null;
        for(let i = 0; i < layers.length; i++) {
            let layer;
            if(layers[i].type === "input") {
                layer = new Layer(prevLayer, "input", layers[i].size, activationFunctions.sigmoid);
                prevLayer = layer;
            } else {
                layer = new Layer(prevLayer, "hidden", layers[i].size, activationFunctions.sigmoid);
                prevLayer = layer;
            }
            this.layers.push(layer);
        }
    }

    // GETTERS
    getLayers() {return this.layers;}

    //HELPER METHODS

    train(data) {
        for(let iter = 0; iter < 500; iter++) {
            for(let i = 0; i < data.length; i++) {
                const predicted = this.predict(data[i].input);
                console.log(`Iteration (${iter}): COST = ${this.cost(predicted, data[i].output)}`);
                const layer = this.getLayers()[this.getLayers().length - 1];
                layer.backPropogate(costPartialDerivatives(predicted, data[i].output));
            }
            for(let i = 1; i < this.getLayers().length; i++) {
                this.getLayers()[i].adjustWeights(data.length);
            }
        }
    }

    predict(inputs) {
        for(let i = 0; i < this.getLayers()[0].getNeurons().length; i++) {
            this.getLayers()[0].getNeurons()[i].setValue(inputs[i]);
        }

        const predicted = [];

        for(let i = 0; i < this.getLayers()[this.getLayers().length-1].getNeurons().length; i++) {
            let pred = this.getLayers()[this.getLayers().length-1].getNeurons()[i].calculateValue();
            //console.log("Output Layer: " + i + " : " + pred);
            predicted.push(pred);
        }

        return predicted;
    }

    cost(predicted, actual) {
        let cost = 0;
        for(let i = 0; i < predicted.length; i++) {
            cost += Math.pow(predicted[i] - actual[i], 2);
        }
        return cost;
    }
}

const myNN = new NeuralNetwork([{type : "input", size : 1},{type : "hidden", size : 2},{type : "hidden", size : 1}]);

myNN.train([
    {input : [-100], output : [0]},
    {input : [100], output : [1]},
   ]);

console.log(`Predicted : ${myNN.predict([-100])}, Actual : 0`);
console.log(`Predicted : ${myNN.predict([100])}, Actual : 1`);