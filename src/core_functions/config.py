# initialize the initial learning rate, number of epochs to train for,
# and batch size
from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only
Path(__file__).parent.parent.absolute()
env_path = Path(__file__).parent.parent.absolute() / '.env'
load_dotenv()
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
