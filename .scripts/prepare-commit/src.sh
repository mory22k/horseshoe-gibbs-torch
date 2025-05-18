# Checks for changes in the ./src directory and prepares a commit if changes are detected.
if git diff --cached --quiet -- ./src; then
  echo "No changes detected in ./src"
else
  echo "Changes detected in ./src"
  task commit:prepare:src
fi
