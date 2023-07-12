# Satisfactory Production Chain Optimizer

[Satisfactory](https://www.satisfactorygame.com/) is a first person factory automation game by [Coffee Stain Studios](https://www.coffeestainstudios.com/).
This program aids the player to find the optimal strategy to produce desired resources.
It selects and combines between all alternative recipes to find a way to maximize your goal while considering your design constraints.

## How to use it

Given that numpy and scipy is installed the program is run with this command:
```
python src/main.py rules/recipe_power.txt output_graph.dot
```

The result is a [dot graph](https://de.wikipedia.org/wiki/Graphviz) and stored in `output_graph.dot`.

We provide rule sets of satisfactory recipes with and without power.
The power rule set will ensure that the power needs are met by the machines and will derive the appropriate amount of power infrastructure.
This will often create much more complicated production chains as it will try to incorporate the entire nuclear power recipes.

By default, the rule sets will maximize the amount FICSIT sink points.
To specify your own constraint make a copy of an existing rule set, change the goal and add your own constraints.

### Input Language
The rule set are specified in the language specified by this grammar.

```
S ::= (Recipe | Constraint | Goal)+
Recipe ::= Expr -> Expr '\n'
Constraint ::= Expr cOp Expr '\n'
cOp ::= '<=' | '==' | '>='
Goal ::= gOp Expr '\n'
gOp ::= 'min' | 'max'
Expr ::= Atom (tOp? Expr)*
tOp ::= '+' | '-' | '/' | '*'
Atom ::= Constant | Variable | '(' Expr ')' | -Atom
Constant ::= \d+(\.\d*)?
Variable ::= \w[A-z ]*\w | \w
```

Additional there are these semantic rules:
- There is exactly a single goal.
- All expressions must simplify to linear expressions.
- The tOp is only optional if multiplying a constant with a variable or expression in parentheses.

Line comments are started with //, block comments start with /* and end with */.
Block comments can be nested.

## How it works

We encode the Satisfactory recipes, resource limitation and optimization goal as a linear program.

Every resource (iron ore, wire, turbo motors, power, ...) is represented by a variable which stores how many parts of that resource are generated per minute.
Additionally, every input and output of a recipe is its own unique variable.
The sum of all output variables for the same resource must equal that resources pool variable.
The same applies to all input variables for the same resource.

Here's a small example:

This recipe using Iron Ore and maximizing the sum of Iron Rods and Iron Plates would be encoded the following

```
30 Iron Ore -> 30 Iron Ingot
30 Iron Ingot -> 15 Iron Rod
30 Iron Ingot -> 20 Iron Plate

max Iron Rod + Iron Plate
```

$$
\begin{align}
30\*IO^{in}\_0 &= 30 \* II^{out}\_{0} \\
15\*II^{in}\_0 &= 30 \* IR^{out}\_{0} \\
20\*II^{in}\_1 &= 30 \* IP^{out}\_{0}\\
IO^{in} &= IO^{pool} \\
II^{in}\_0 + II^{in}\_1 &= II^{pool} \\
II^{out}\_0 &= II^{pool} \\
IR^{out}\_0 &= IR^{pool} \\
IP^{out}\_0 &= IP^{pool} \\
max\quad I^{pool}
\end{align}
$$

Additionally, all variables are forced to be positive.

Besides recipes the input format supports arbitrary linear equalities and inequalities.
These can be specified as an arbitrarily nested arithmetic formula (as long as they remain linear).

## Encoding Satisfactory Rules

The rules for all renewable recipes are found [here](./rules).
For the most part the translation is fairly straightforward, however there are a few complications.

### Resource limitations

The amount of raw resources one can gather per minute is strictly limited by number and quality of resource nodes.
These limitations are specified by simple inequalities.

### Overclocking

Almost all machines can be overclocked in Satisfactory.

For machines that consume power, their power consumption given a certain overclocking factor is determined by this [formula](https://satisfactory.wiki.gg/wiki/Clock_speed)

$$power usage = initial power usage \* clock speed^{log_2(2.5)}$$

This is especially relevant for resource extractors like miners.
By overclocking them we get more resources to but need to pay exponentially more power.
We can not model the non-linearity of the power formula in a linear program.
But we would like our optimizer to make trade-offs by to use less power of if it does not need all resources.
Therefore, we conservatively approximate the costs of a resource extractor by determining the maximum power used by an extractor if overclocked at 2.5.
For Miners MK.3 on pure resource nodes it does not make sense to overclock them at 2.5 as they are limited by the outgoing conveyor belt to 780 items per minute. Theoretically they are able to produce 1200 items per minute.
Indeed, they should only be overclocked at 1.625 for maximum efficiency.
This is why pure nodes have a different power requirement in the [rule set](./rules/recipes_power.txt).

For power consumers we assume they are never overclocked/underclocked.
Technically, one could save power by replacing a normal constructor with $n$ constructors where each is under clocked at $\frac{1}{n}$.
At the limit the power costs would converge to zero.
For practical gameplay purposes however this would be exceedingly tedious, so we ignore this possibility.

### Particle Accelerators

Particle Accelerators need a variable amount of power when active.
We do not try to model that and instead use the average amount needed by the recipes.
Any variability in power consumption needs to be balanced by adding an appropriate amount of power storage.
