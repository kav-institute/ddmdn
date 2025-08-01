#### Build image from scratch
Define parameters and paths in user.env to map the correct code and data paths as well as the correct user rights to the container. A Nvidia GPU is highly recommended.

```bash
# Directory mapping:
REPOS_PATH='/<YOUR_PATH_TO_THE_REPOSITORY>' ---> Is mapped in the container to: /workspace/repos
DATA_PATH='/<YOUR_PATH_TO_THE_DATA>' ---> Is mapped in the container to: /workspace/data

# Example:
REPOS_PATH='/mnt/user123/ddmdn/data' # in which the preprocessed data and pretrained trained_models are located
DATA_PATH='/mnt/user123/ddmdn/repos' # in which the ddmdn github repository is cloned to
```

```bash
docker compose --env-file user.env build --no-cache
```


#### Run container from image
```bash
docker compose --env-file user.env build up -d
```


#### Shutdown container
```bash
docker compose --env-file user.env build down
```