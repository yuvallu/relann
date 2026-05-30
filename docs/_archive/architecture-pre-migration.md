# ParENT Archetecture doc

This document outlines a possible archetecture for the ParentNN model,
as an bi directional embedded DSL in python whose target language is pytorch and hopefully a non python runtime like torchscript etc.
This document assumes familiarity of the ParENT model from the formal proposal.

We will namespace parent concepts with `ent` as short hand to distinguish them from overloaded concepts from the pytorch ecosystem (namespaced via `nn`).
The PareENT concepts that we will implement here are:
* `ent.Module` A single conjunctive derivation rule of embedded tuples for exisitng embedded tuples
    * This is the most basic statement in the ParENT model
* `ent.Model` Also known as an SSM, a sequence of Modules used to compose logic 
    * The analogy of a function in the ParENT model
* `ent.Program` a full program.



TODO

## Codebase flow

There are two several flows for the user

1. Parsing a `ent.Model` into a torch `nn.Module`
    * Parse the model into a differential embedded relational algebra (DE-RA) term graph 
    * Transpiling the DE-RA term graph into a torch module
    * saving both the term graph and the torch module into a registry in the ParENT library
2. Compile a program
    * Parse all modules into torch modules and dataloaders
    * return a Program class that can be executed and queried
        * for execution see next point
        * for querying, we can query the state of intrinsic relations, getting an embedded relation as a result.
3. Execute a program
    * run a compiled program line by line
  

## User Interface

The current idea to embedd ParENT in python is to emulate the embedding approach found in [LMQL](https://lmql.ai/docs/lib/python.html)

The idea is as follows:

* We will implement two decorators, that will take a function whose docstring contains the ParENT code that describes it.
* The decorator will parse the docstring and return the object that it was transpiled to.
* transformations will be either 
    * imported from the global namespace
    * fed to the function via a named kwarg 

for example, immitating the example in the moonshot proposal:

```prolog
% Step 1: We define a GNN like regression model
model SepsisRisk(Patients:(4)⟨0⟩,VitalSigns(5)⟨0⟩) -> (1)⟨1⟩:
    % Step 1.1: Embedding Patient and Vital Signs based on relational columns
    PatientEmbedding(P)⟨[A,S],Linear(2,8)⟩ :- Patients(P,_,A,S).
    VitalSignsEmbedding(P,R)⟨[T,H,R],Linear(3,8)⟩ :- VitalSigns(P,T,H,_,R).
    
    % Step 1.2: Aggregate embeddings of vital signs per patient
    VitalSignsAggregated(P)⟨AVG(z)⟩ :- VitalSignsEmbedding(P, R)⟨z⟩.

    % Step 1.3: Convolve the Patient embedding with the aggregate embedding of her vital signs.
    % Patient and VitalSigns embeddings are matched via join.
    PatientVitalEmbedding(P)⟨Concat(z1,z2),Linear(16,32),ReLU⟩ :- PatientEmbedding(P)⟨z1⟩, VitalSignsAggregated(P)⟨z2⟩.

    % Step 1.4: Compute a regression score via a linear projection
    Out(P)⟨Linear(32,1)⟩ :- PatientVitalEmbedding(P)⟨z⟩.

% Step 2: Define train and test split of our data
TrainP(P,N,A,S) :- Patients(P,N,A,S), PatientSplit(P,'train').
TrainV(P,T,H,N,R) :- VitalSigns(P,T,H,N,R),PatientSplit(P,'train').
TestP(P,N,A,S) :- Patients(P,N,A,S), PatientSplit(P,'test').
TestV(P,T,H,N,R) :- VitalSigns(P,T,H,N,R),PatientSplit(P,'test').

% Step 3: Define a loss function: Mean Squared Error between the prediction (model SepsisRisk) and the ground truth (relation GroundTruth)
Loss()⟨MSE(z,Label)⟩ :- SepsisRisk{TrainP,TrainV}(P)⟨z⟩, GroundTruth(P,Label).

% Step 4: Train the model by the loss function Loss
Fit By Loss()⟨z⟩.

% Step 5: Predict the risk for the test set using our trained model
Prediction(P,⟨z⟩) :- SepsisRisk{TestP,TestV}(P)⟨z⟩.

```

Will look like so:

```python

# here the inputs and outputs are given by type hints, not sure about that but could be an option
# Here Liner and Relu are assumed to be in the global namespace
@ent.Model
def SepsisRisk(Patients:ent.Rel[4,0],VitalSigns:ent.Rel[5,0])->ent.Rel[1,1]:
    '''ent
    # comments are allowed in the docstrings and are ignored
    PatientEmbedding(P)⟨[A,S],Linear(2,8)⟩ :- Patients(P,_,A,S).
    VitalSignsEmbedding(P,R)⟨[T,H,R],Linear(3,8)⟩ :- VitalSigns(P,T,H,_,R).
    VitalSignsAggregated(P)⟨AVG(z)⟩ :- VitalSignsEmbedding(P, R)⟨z⟩.
    PatientVitalEmbedding(P)⟨Concat(z1,z2),Linear(16,32),ReLU⟩ :- PatientEmbedding(P)⟨z1⟩,
        VitalSignsAggregated(P)⟨z2⟩.
    Out(P)⟨Linear(32,1)⟩ :- PatientVitalEmbedding(P)⟨z⟩.
    '''

# we can also imagine passing them explicitly
tfms = {'Liner':...,'ReLU':...}

@ent.Model(tfms=tmfs)
def SepsisRisk(Patients:ent.Rel[4,0],VitalSigns:ent.Rel[5,0])->ent.Rel[1,1]:
    ...


@ent.Program
def PredictSepsis():
    '''ent
    # here, the SepsisRisk model is recognized, since it exists in the registy
    ...
    '''

```
We can also optionally have the ent.Model/ent.Program accept a string rather than a function defining the parent code that they map to, to enable users to dyanmically construct ParentCode

```python

code=''
for i in range(10):
    code+=f'''ent
    Layer_{i}(P):-...
    '''

SepsisRisk = ent.Model(code,
    inputs_schema={'Patients'[4,0],'VitalSigns'[5,0]}
    output_schema=[1,1]
    )

```

Note that we will implement a VSCODE extension like LMQL, that will cause the docstring that begin with `ent` in python source code to be highlighted using the syntax highlighting of our ParENT language.

## Parse models/Programs

Parsing will be written via the [Lark](https://github.com/lark-parser/lark)
Semantic checks can be written manually via a lark graphVisitor, or can be done via micropass archetecture like in Spannerlib.

The parsing will turn both Models and Programs into Naive dataclass representations that look somthing like the following

```python

@dataclass
class Module():
    name: str
    lhs_rel:tuple(Str)
    ...
    ...

class Program():
    statements: List[Union(Statement|FitStatement)]

```

## Parse models/modules to DE-RA trees

The dataclasses of a module/model can now be converted to a `nx` graph that models a DE-RA algebra.
The semantics of the algebra is as follows:

* each node, when materialized, returns an multi-embedded relation
    * a dataframe and a tuple of tensors
    * its a tuple of tensors and not 1 tensor to support the semantics of our join operation
* the algebra has the following operators with the following semantics
  * union<agg>
    * takes the union of the relations, and merge embeddings of duplicate lines via agg
  * difference
    * difference of relations, carry the embeddings
  * product
    * product of relation, embedding of each row is the horzontal concatenation of the embedding tuple of each original row that made up the product row.
  * project<agg>
    * projects the relation on the given columns. embeddings of rows that are duplicates after projection are merged via agg
  * selection<theta>
    * selects rows of relations, embeddings are carried
  * join
    * join of relations, embedding of each row is the the horzontal concatenation of the embedding tuple of each original row that made up the join row.
  * embedding_map<T>
    * for our embedded relation $(R,(E_1,...,E_n))$, R remains intact, T is mapped over rows of $(E_1,...,E_n)$, for a differential transformation T
    * in practice, we expect T to be a transformation that is broadcasted across the first axes, allowing us to simply map T over $(E_1,...,E_n)$ to get the new embedding tuple.
    * We will want to understand whether T is a parameterless/frozen transformation, so that we can take advantage of it during caching.
  * agg<group_attr,relation_aggs,embedding_agg>
    * group relations based on group attributes which are a set of columns
    * optionally aggregate column attributes using relational aggs (mean,avg,...) on columns that are not in group_attr
    * aggregate groups of embeddings used the embedding_agg function.


## Implementing RA opertions

Since we can, we should probably have both the content and the embedding be in GPU
We can run relational operations on the content via something like [RAPIDS cuDF](https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/)

Each RA operation should be translated into an EmbeddedRelation `ent.eRel` or some such name.
This class will be a subclass of `nn.Module` with the following interface:


```python

class EmbeddableRelation(nn.Module):
    def instantiate(self,*eRels):
        # Here we compute the content, of the new eRel, which should remain unchanged throughout forward passes, which only change the embeddings of the relation.
        # here, stuff like the join keys are created and stored for shutlling the embedded vectors of the inputs to the correct places in the output

    def forward(self,*eRels):
        # this method assumes self has been instantiated already
        # and only changes the embeddings

```


## Transpile models/modules to torch.nns

Whenever we need to construct a torch module out of a module/model, we create a torchmodule dyanmically that composese our RA operations according to the RA graph.

For repeating patterns that can be better optimized, we can create "fused" operators.
These fused operators can basically implement a specific subgraph more effectively.
This is how we do physical optimizations.

Logical optimizations are rewritting of the RA graph.

Once we have physical optimizations, we need to materilize the nn.Module in a way that chooses which subgraphs will be instantiated by the fused operators.

in GPU as soon as it is loaded.


## Parse extensional assignments into torch SQL dataloaders

Basically, we should make data loaders, maybe disk cached, for modules or clauses happening within the main scope of an `ent.Progam` whose terms are completely extensional relations.
These dataloaders should be "SQL dataloaders" they are instantiated with an SQL query which we generate based on the rule (could also be supplied by a user).

They take this SQL query and make a paginated query to the DB, the size of page is the minibatchsize given to the dataloader.

## Running Fit and infer statement

Fit and infer statements, are turned into our custom training and inference loops.
They will look something like this:

```python

# we only includes the details where things are different from regular torch

# happens before

# we can have frozen induced subtrees cached (for example, bert embedding of textual columns)
net,loss_fn = transpile_target_to_nn_Module()
opt = ...
#training
for epoch in epochs:
    for batch in batches:
        opt.zero_grad()
        for minibatch in batch:
            data = data_loader.next()
            labels = lable_loader.next()
            # this could be cached since it wont change between epochs
            net.instantiate(data)
            loss = loss_fn(net.forward(data),labels)

        optimizer.step()

#inference

for batch in batches
    data = data_loader.next()
    net.instantiate(data)
    out = net.forward(data)

```


## Transpiling to TorchScript/or other lowlevel runtime

We would need to make sure the nn.Module program we transpile is traceable by the lowlevel run time we want to use.
