
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Provide a game env as argument"

    arg = sys.argv[1]
    
    game_list = [
                 "pursuit", 
                 "pursuitTweak" 
                ]
    if arg not in game_list:
        raise Exception("Input a valid game. Choose from {}".format(game_list))

    if arg == "pursuit":
        import sisl_games.pursuit.test
    
    if arg == "pursuitTweak":
        import sisl_games.pursuitTweak.test
