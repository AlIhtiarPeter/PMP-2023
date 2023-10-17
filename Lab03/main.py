# This is a sample Python script.
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

model = BayesianNetwork([('C','I'),('C','A'),('I','A')])

CPD_C = TabularCPD(variable="C",variable_card=2,values=[[0.9995],[0.0005]])
CPD_I = TabularCPD(variable="I",variable_card=2,values=[[0.99,0.97], [0.01,0.03]], evidence=['C'], evidence_card=[2])
CPD_A = TabularCPD(variable="A",variable_card=2,values=[[0.9999,0.05,0.98,0.02],[0.0001, 0.95, 0.02, 0.98]],
                   evidence=['C', 'I'], evidence_card=[2, 2])
print(CPD_C)
print(CPD_A)
print(CPD_I)

model.add_cpds(CPD_A,CPD_I,CPD_C)

print(model.check_model())

inference = VariableElimination(model)

c = inference.query(variables=['C'])
i = inference.query(variables=['I'])
a = inference.query(variables=['A'])

print(c)
print(i)
print(a)

inference = VariableElimination(model)
ex2 = inference.query(['C'], evidence={'A': 1})
print(ex2)
ex3 = inference.query(['I'], evidence={'A': 0})
print(ex3)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()
