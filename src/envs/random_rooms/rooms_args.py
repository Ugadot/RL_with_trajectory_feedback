import argparse

def get_random_rooms_arg_parser():

    parser = argparse.ArgumentParser(add_help=False)
    env_group = parser.add_argument_group("rooms env args")
#   parser.add_argument(
#       '--add-punishment',
#       action='store_true',
#       default=False,
#       help='add punishment to sparse reward')
    env_group.add_argument(
        '--rooms',
        default=4,
        type=int,
        help='Use rooms in env (if 0 - no rooms), default=4')
    env_group.add_argument(
        '--const-rooms',
        action='store_true',
        default=False,
        help='Keeps env rooms constant - rooms dont change')
    env_group.add_argument(
        '--const-goal',
        action='store_true',
        default=False,
        help='keeps env constant - goal doesnt change')
    env_group.add_argument(
        '--obs-type',
        choices=['flat', 'original', 'reduced'],
        help='Type of observation to use (flat, original or reduced), default="original"',
        default='original')
    env_group.add_argument(
        '--rows',
        type=int,
        default=10,
        help='Random rooms map size - rows, default=10')
    env_group.add_argument(
        '--cols',
        type=int,
        default=10,
        help='Random rooms map size - cols, default=10')
    env_group.add_argument(
        '--max-steps-fact',
        type=float,
        default=3,
        help='Random rooms - max steps = (rows + cols) * fact, default fact=3')
    env_group.add_argument(
        '--inner-counter-in-state',
        type=bool,
        default=False,
        help='Random rooms - choose if to use 4th observation channel - history')
    env_group.add_argument(
        '--n-inner-resets',
        type=int,
        default=1000,
        help='Random rooms - n episodes of hard reset - changes walls, goal if not const, default=1000')
    env_group.add_argument(
        '--far_from_goal',
        default=False,
        help='Make agent as far from goal as possible',
        action='store_true')
    return parser
