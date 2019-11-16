# ECE 276C HW4

Mingwei Xu A53270271

## Run the codes

### Question 1 DDPG

To train the model, run

```bash
python ddpg_reach.py
```

It will train the model and save the actor network as `ddpg_actor.pkl` for evaluation.

To evaluate the policy, run

```bash
python ddpg_reach.py --test
```

It will load `ddpg_actor.pkl` and test it in randomly initialized environment.

### Question 2 TD3

To train the model, run

```bash
python td3_reach.py
```

It will train the model and save the actor network as `td3_actor.pkl` for evaluation.

To evaluate the policy, run

```bash
python td3_reach.py --test
```

It will load `td3_actor.pkl` and test it in randomly initialized environment.

## Tests

To test the codes, do `cd tests/` and run `pytest`