'use strict';

/*
    GLOBAL VARIABLES, OBJECTS, AND FUNCTIONS
*/

const learningRate = 0.4;

const activationFunctions = {
    none : {
        fn : function(num) {
            return num;
        },
        derivative : function (num) {
            return 1;
        }
    },
    relu : {
        fn : function(num) {
            if (num <= 0) { return 0; }
            else { return num; }
        },
        derivative : function(num) {
            if (num <= 0) { return 0; }
            else { return 1; }
        }
    },
    sigmoid : {
        fn : function(num) {
            return (1 / (1 + Math.pow(Math.E, (-1) * num)));
        },
        derivative : function(num) {
            return (Math.pow(Math.E, (-1) * num) / Math.pow(1 + Math.pow(Math.E, (-1) * num), 2));
        }
    }
}

function generateRandomWeights(size) {
    let arr = [];
    for (let i = 0; i < size; i++) {
        arr.push(8 * Math.random() - 4);
    }
    return arr;
}

function generateRandomBias() {
    return 2 * Math.random() - 1;
}

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

class Neuron { constructor() { } }

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

    // SETTERS


    // HELPER METHODS

    // Calculates the value of a neuron recursively and stores it, as well as the zeta value of the neuron. Returns the calculated value
    calculateValue() {
        this.calculatedZeta = 0;
        for (let i = 0; i < this.previousLayer.getNeurons().length; i++) {
            this.calculatedZeta += this.weights[i] * this.previousLayer.getNeurons()[i].calculateValue();
        }
        this.calculatedZeta += this.bias;
        this.calculatedValue = this.getActivation().fn(this.calculatedZeta);
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
        this.input = this.activationFunction.fn(input);
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

    getPreviousLayer() { return this.previousLayer; }
    getNeurons() { return this.neurons; }
    getSize() { return this.size; }
    getActivationFunction() { return this.activationFunction; }

    // HELPER METHODS
    backPropogate(partialDerivatives) {
        if (this.type === "hidden") {
            for (let i = 0; i < this.getNeurons().length; i++) {
                // Change weights
                for (let j = 0; j < this.getNeurons()[i].weights.length; j++) {
                    this.getNeurons()[i].weightChanges[j] += (learningRate) * (this.getPreviousLayer().getNeurons()[j].getCalculatedValue()) * (this.getActivationFunction().derivative(this.getNeurons()[i].getCalculatedZeta())) * (partialDerivatives[i]);
                }
                // Change bias
                this.getNeurons()[i].biasChange += (learningRate) * (this.getActivationFunction().derivative(this.getNeurons()[i].getCalculatedZeta())) * (partialDerivatives[i]);
            }
            // For each neuron in the previous layer, add the current layer's partial derivatives for the neuron. Store in an array
            const pDNew = [];
            for (let i = 0; i < this.getPreviousLayer().getNeurons().length; i++) {
                let pDCurrent = 0;
                for (let j = 0; j < this.getNeurons().length; j++) {
                    pDCurrent += (learningRate) * (this.getNeurons()[j].getWeights()[i]) * (this.getActivationFunction().derivative(this.getNeurons()[j].getCalculatedZeta())) * (partialDerivatives[j]);
                }
                pDNew.push(pDCurrent);
            }
            this.getPreviousLayer().backPropogate(pDNew);
        }
    }

    adjustWeights(dataSize) {
        for (let i = 0; i < this.getNeurons().length; i++) {
            for (let j = 0; j < this.getNeurons()[i].getWeights().length; j++) {
                this.getNeurons()[i].weights[j] += this.getNeurons()[i].weightChanges[j] / dataSize;
                this.getNeurons()[i].weightChanges[j] = 0;
            }
        }
    }
}

class NeuralNetwork {
    constructor(iterations, layers) {
        this.iterations = iterations;
        this.layers = [];
        let prevLayer = null;
        for (let i = 0; i < layers.length; i++) {
            let layer;
            if (layers[i].type === "input") {
                layer = new Layer(prevLayer, "input", layers[i].size, layers[i].activation);
                prevLayer = layer;
            } else {
                layer = new Layer(prevLayer, "hidden", layers[i].size, layers[i].activation);
                prevLayer = layer;
            }
            this.layers.push(layer);
        }
    }

    // GETTERS
    getLayers() { return this.layers; }
    getIterations() { return this.iterations; }

    //HELPER METHODS

    train(data) {
        for (let iter = 0; iter < this.getIterations(); iter++) {
            let iterCost = 0;
            for (let i = 0; i < data.length; i++) {
                const predicted = this.predict(data[i].input);
                iterCost += this.cost(predicted, data[i].output) / data.length;
                const layer = this.getLayers()[this.getLayers().length - 1];
                layer.backPropogate(costPartialDerivatives(predicted, data[i].output));
            }
            console.log(`Iteration ${iter + 1}: COST = ${iterCost}`);
            for (let i = 1; i < this.getLayers().length; i++) {
                this.getLayers()[i].adjustWeights(data.length);
            }
        }
    }

    predict(inputs) {
        for (let i = 0; i < this.getLayers()[0].getNeurons().length; i++) {
            this.getLayers()[0].getNeurons()[i].setValue(inputs[i]);
        }

        const predicted = [];

        for (let i = 0; i < this.getLayers()[this.getLayers().length - 1].getNeurons().length; i++) {
            let pred = this.getLayers()[this.getLayers().length - 1].getNeurons()[i].calculateValue();
            predicted.push(pred);
        }

        return predicted;
    }

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

const myNN = new NeuralNetwork(10000, [{ type: "input", size: 3, activation: activationFunctions.none }, { type: "hidden", size: 6, activation: activationFunctions.sigmoid }, { type: "hidden", size: 3, activation: activationFunctions.sigmoid }]);

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

console.log(`Predicted : ${myNN.predict([1, 0, 0])}, Actual : 0, 1, 1`);
console.log(`Predicted : ${myNN.predict([0, 0, 1])}, Actual : 1, 1, 0`);


/*
myNN.train([
    { input : [0.03, 0.7, 0.5], output : [0] },
    { input : [0.16, 0.09, 0.2], output : [1] },
    { input : [0.5, 0.5, 1.0], output : [1] }
]);

console.log(`Predicted : ${myNN.predict([1, 0.4, 0])}, Actual : 1`);
*/