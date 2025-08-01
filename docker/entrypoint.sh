#!/bin/bash
echo "--- Setting up container, please wait..."
echo "--- Now as root..."

su - root <<!
root
service ssh start
setfacl -R -d -m u::rwx,g::rwx,o::rwx /workspace/repos
setfacl -R -d -m u::rwx,g::rwx,o::rwx /workspace/data
setfacl -R -m u::rwx,g::rwx,o::rwx /workspace/repos
setfacl -R -m u::rwx,g::rwx,o::rwx /workspace/data
!

echo "--- End of root..."
echo "--- Additional installations..."
# None here

echo "--- Additional tasks..."
# None here

echo "--- Additional container setup completed, ready for work..."
tail -F /dev/null