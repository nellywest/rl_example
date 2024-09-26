## Prerequisites

Install kubectl (>= 1.23), Helm (>= v3.4), Kind, and Docker, as described in https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/raycluster-quick-start.html.


## Run code in a python environment

1. Create the environment:

```python3 -m venv env```

2. Activate it:

```source env/bin/activate```

3. Install dependencies:

```pip install -r requirements.txt```

4. Run file:

```python3 <file_name>```


## Run code in a docker container

1. Build docker image:

```docker build -t ray-cluster .```

2. Start the container in interactive mode:

```docker run --cpus="20" --gpus all --shm-size=15gb -it ray-cluster```

3. Copy the file into the container:

```docker cp <file_name> <container_id>:/app```

4. Run the file in the container:

```docker exec -w /app <container_id> python3 <file_name>```


## Run code in Kubernetes:

TO DO: Make GPU accessible in Kubernetes

1. Rename the image:

```docker tag ray-cluster <user_name>/ray-cluster```

2. Log in to the registry:

```docker login```

3. Push the image to the registry:

```docker push <user_name>/ray-cluster```

4. Update image name in raycluster.yaml

5. Create a Kubernetes cluster:

```kind create cluster --image=kindest/node:v1.26.0```

6. Deploy the KubeRay operator with the Helm chart repository.

```
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1
```

7. Start a KubeRay cluster:

```kubectl apply -f raycluster.yaml```

8. Store head pod name in a variable:

```export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)```

9. Run a file in the pod:

```cat <file_name> | kubectl exec -i $HEAD_POD -n default -- python```