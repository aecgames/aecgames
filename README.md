# Agent-Environment Cycle (AEC) Games

* To run `pursuit`, use the command `python3 main_run.py pursuit`.
* To run `pursuitTweak` (a tweaked version of pursuit), use the command `python3 main_run.py pursuitTweak`.

## Requirements
This repo uses the Reinforcement Library toolkit `RLlib` from [Ray](https://github.com/ray-project/ray). This specific wheel of the `Ray` package needs to be installed: \url{link to Ray wheel}. Once the wheel is downloaded, install it as `pip install -U path/to/wheel.whl`. Install the other required packages using the command `pip install -r requirements.txt`. Required Python version is `3.7.6`.


## Learning the games
* Run `python3 parameterSharingPursuit.py RLmethod` to train an RL method `RLmethod` (e.g. PPO) on `pursuit`.
* Run `python3 parameterSharingPursuitTweak.py RLmethod` to train an RL method `RLmethod` (e.g. PPO) on `purstuitTweak`.

## Playing the learned games
Use `play_pursuit.py` and `play_pursuitTweak.py` to re-play the games using the learned RL policies. Simply change the `checkpoint_path` variable and make sure `params.pkl` is in the correct place, relative to the former.

## Note on log files/logging/plotting

* RLlib has many interesting and unresolved bugs regarding logging episode reward vs policy reward, and them sometimes being off by a factor of the number of agents. This is why the data is sometimes scaled in the files which generate plots. We verified we scaled them correctly by playing the trained policies and adding up typical rewards ourselves in each case.
