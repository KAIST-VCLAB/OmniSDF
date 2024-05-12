docker build \
--build-arg PHYSICAL_UID=$(id -u) --build-arg PHYSICAL_GID=$(id -g) -t omnisdf:1.0 .