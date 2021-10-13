import os
import pickle
import sys
master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, master_dir)

from src.utils.user_interface import saved_data_render, Segment


def rate_sequences(sequence_dir):
    if not os.path.exists(sequence_dir):
        print ('Directory does not exists')
    else:
        rated_sequences_dir = os.path.join(sequence_dir, "rated")
        os.makedirs(rated_sequences_dir, exist_ok=True)
        for filename in os.listdir(sequence_dir):
            print ("Filename-", filename)
            if filename.endswith(".pkl"):
                rated_sequence_path = os.path.join(rated_sequences_dir, f"rated_{filename}")
                if not os.path.exists(rated_sequence_path):
                    with open(os.path.join(sequence_dir, filename), "rb") as PKL_FILE:
                        sequence = pickle.load(PKL_FILE)
                   
                    user_result = -1
                    while(user_result < 0 or user_result > 10):
                        print("Agent results are:")
                        images = sequence.frames
                        saved_data_render(images)
                        try:
                          user_result = int(input("How the user performed?"))
                        except:
                          print("Enter integer between 0 and 10 please")
                          user_result = -1

                    sequence.change_user_reward(user_result / 10.0)

                    with open(rated_sequence_path, "wb") as PKL_FILE:
                        pickle.dump(sequence, PKL_FILE)

if __name__=="__main__":
    sequence_dir = sys.argv[1]
    rate_sequences(sequence_dir)

