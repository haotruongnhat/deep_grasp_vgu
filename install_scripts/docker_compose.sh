#!/bin/bash

### Install docker-compose
### https://docs.docker.com/compose/install/

sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

### Testing
docker-compose --version