## Import modules

import csv
import random
import operator
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Load training and testing data

def load_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({
                'sales': float(row['sales']),
                'price': float(row['price']),
                'discount': float(row['discount']),
                'advertisement': float(row['advertisement']),
                'cost': float(row['cost'])
            })
    return data

train_data = pd.read_csv(r"sample_data_for_training.csv") ## Add file path to training data csv file here
test_data = pd.read_csv(r"sample_data_for_testing.csv")  ## Add file path to training data csv file here


# Genetic programming settings
individual_size = 50
population_size = 500
generations = 15
toolbox = base.Toolbox()

pset = gp.PrimitiveSet("MAIN", 4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.truediv, 2)

# Fields
toolbox.register("attr_price", random.uniform, -10, 10)
toolbox.register("attr_discount", random.uniform, -10, 10)
toolbox.register("attr_advertisement", random.uniform, -10, 10)
toolbox.register("attr_cost", random.uniform, -10, 10)

# Operations
toolbox.register("add", operator.add)
toolbox.register("sub", operator.sub)
toolbox.register("mul", operator.mul)
toolbox.register("div", operator.truediv)

# Function set
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Fitness function
def eval_sales(individual, data):
    fitness = 0
    for sales, price, discount, advertisement, cost in zip(data['sales'], data['price'], data['discount'], data['advertisement'], data['cost']):
        code = compile(str(individual), '<string>', 'eval')
        globals_dict = {}
        locals_dict = {
            'truediv': lambda x, y: x / (y + 1e-9),
            'sub': operator.sub,
            'add': operator.add,
            'mul': operator.mul,
            'ARG0': price,
            'ARG1': discount,
            'ARG2': advertisement,
            'ARG3': cost
        }
        try:
            prediction = eval(code, globals_dict, locals_dict)
        except Exception as e:
            print("Error evaluating individual: ", e)
            prediction = 0
        fitness += (sales - prediction) ** 2
        # Add a penalty term for solutions that only use ARG0
        if 'ARG1' not in str(individual) and 'ARG2' not in str(individual) and 'ARG3' not in str(individual):
            fitness += 100
    return (fitness,)


# Register the fitness function
toolbox.register("evaluate", eval_sales, data=train_data)

# Register the mate, mutate, and select functions
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=100) ## Edit 'tournsize' to specify the tournament size 


def visualize_data(data, title):
    plt.scatter(data['price'], data['discount'], c=data['sales'])
    plt.xlabel('Price')
    plt.ylabel('Discount')
    plt.title(title)
    plt.show()

def visualize_predictions(actual, predicted, title):
    plt.plot(actual, label='Actual Sales')
    plt.plot(predicted, label='Predicted Sales')
    plt.xlabel('Index')
    plt.ylabel('Sales')
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    random.seed(64)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)
    
    ## Edit here to set crossover, mutation annd elite probability
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.1, ngen=generations, stats=stats, halloffame=hof, verbose=True)

    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    # print ('hof is ', hof)

# Predict sales using the best individual on test_data
def predict_sales(best_individual, data):
    predictions = []
    for price, discount, advertisement, cost in zip(data['price'], data['discount'], data['advertisement'], data['cost']):
        code = compile(str(best_individual), '<string>', 'eval')
        globals_dict = {'truediv': lambda x, y: x / (y + 1e-9), 
                        'sub': operator.sub, 
                        'add': operator.add, 
                        'mul': operator.mul}
        locals_dict = {'ARG0': price, 'ARG1': discount, 'ARG2': advertisement, 'ARG3': cost}
        try:
            prediction = eval(code, globals_dict, locals_dict)
        except Exception as e:
            print("Error predicting sales: ", e)
            prediction = 0
        predictions.append(prediction)
    return predictions


# Predict sales 
predictions = predict_sales(hof[0], test_data)
actual_sales = test_data['sales']


# Compare predicted sales with actual sales in test_data
mse = ((actual_sales - predictions) ** 2).mean()
r2 = r2_score(actual_sales, predictions)
mae = np.mean(np.abs(actual_sales - predictions))
rmse = np.sqrt(np.mean((actual_sales - predictions) ** 2))
mpe = np.mean(np.abs((actual_sales - predictions) / actual_sales)) * 100
mape = np.mean(np.abs((actual_sales - predictions) / actual_sales)) * 100
rmspe = np.sqrt(np.mean(((actual_sales - predictions) / actual_sales) ** 2)) * 100
cv = np.std(predictions) / np.mean(predictions)


print(f"Mean squared error between predicted and actual sales: {mse}")
print(f"R-squared value between predicted and actual sales: {r2}")
print(f"MAE between predicted and actual sales: {mae}")
print(f"RMSE between predicted and actual sales: {rmse}")
print(f"MPE between predicted and actual sales: {mpe}%")
print(f"MAPE between predicted and actual sales: {mape}%")
print(f"RMSPE between predicted and actual sales: {rmspe}%")
print(f"CV of prediction: {cv}")