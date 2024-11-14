from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent","Starts")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_keypresent = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.70], [0.30]],
    state_names={"KeyPresent":["yes","no"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent":["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

def main():
    query1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print("\n1. Probability that the battery is not working given that the car will not move:\n", query1)
    query2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print("\n2. Probability that the car will not start given that the radio is not working:\n", query2)

    print("\n3. Probability of radio working given battery is working given that the battery is working, before and after discovering car has gas:")
    query3_before = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    print("Before gas discovery:\n", query3_before)
    query3_after = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print("After gas discovery:\n", query3_after)

    print("\n4. Probability of ignition failing given that the car doesn't move, before and after observing car does not have gas:")
    query4_before = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    print("Before gas observation:\n", query4_before)
    query4_after = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print("After gas observation:\n", query4_after)

    query5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print("\n5. Probability that the car starts given that the radio works and it has gas:\n", query5)

    query6 = car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"})
    print("\n6. Probability that the key is not present given that the car does not move:\n", query6)

if __name__ == "__main__":
    main()