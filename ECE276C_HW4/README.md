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

## Tests

To test the codes, do `cd tests/` and run `pytest`