"""Run pytest on a gerui compute node; args are passed to pytest."""
import subprocess, sys
sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-p", "no:cacheprovider"] + sys.argv[1:]))
