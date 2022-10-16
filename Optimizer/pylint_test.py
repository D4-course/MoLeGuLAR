import sys
from pylint.lint import Run

results = Run(sys.argv[1:], do_exit=False)