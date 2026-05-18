this file is an instruction file - a meta program instructor based on which you will be writing the code.

this codebase is going to change the world. it is going to act as an global_codebase for all the AGI systems.


rule 1 : Generic agent architecture/scaffolding: likely allowed.

Examples of questionable/bad scaffolding:
if game name == X, use solver X
hardcoded ARC-AGI-3 level rules
manual lookup tables
source-code-derived game-specific policies

Examples of likely acceptable scaffolding:
world model
memory system
planner
curiosity module
exploration policy
training loop
latent-state model



rule 2 : no hardcoding
rule 3 : do not keep modifying the code to get the output that i asked for, instead of rethink and make the agent learn by itself
rule 4 : this is a learner so every piece or word of code written must be intended to make sure our model keeps learning, by lraning i don't mean code keeps getting bigger and bigger.
rule 5 : the code must be written not to solve current problem but any type of problem. let it be identifying a problem, finding a way out and so on.
rule 6 : do not skip try to indulge in violating any of the above 5 rules just because the coding needs to be completed.

feel free to use pytorch, neural network, CNN or any state of the art models like JEPA, HRL.

For this exercise, the production should fulfil the below and looks like this:

Agents get no instructions and must explore/adapt in novel environments.
Evaluation has no internet access.
Prize-eligible submissions must be open source/reproducible.
The benchmark tests exploration, modeling, goal-setting, planning, and execution.