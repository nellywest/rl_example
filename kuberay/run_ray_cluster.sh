#!/usr/bin/env zsh

# Apply the RayCluster configuration
kubectl apply -f kuberay/raycluster.yaml

wait 120

# Store the head pod name in a variable
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)

# Port forward the web interface from the head pod
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265 &>/dev/null &

# Run the python script on the head pod
cat hello.py | kubectl exec -i $HEAD_POD -n default -- python

# Get the logs from the head pod
kubectl logs $HEAD_POD -n default