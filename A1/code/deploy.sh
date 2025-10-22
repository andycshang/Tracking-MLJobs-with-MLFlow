#!/bin/bash
set -e  # exit if false

PROJECT_NAME="andycshang-mlops-f25"
IMAGE_NAME="fastapi-model"
EC2_USER="ec2-user"
EC2_IP="54.90.229.217"
PEM_KEY="mymodel.pem"
REMOTE_PATH="/home/ec2-user"
PORT="8000"

echo "Step 1/6: Building Docker image..."
docker build -t $IMAGE_NAME:latest .

echo "Step 2/6: Saving Docker image to tar file..."
docker save -o ${IMAGE_NAME}.tar ${IMAGE_NAME}:latest

echo "Step 3/6: Uploading tar to EC2..."
scp -i $PEM_KEY ${IMAGE_NAME}.tar ${EC2_USER}@${EC2_IP}:${REMOTE_PATH}/

echo "Step 4/6: Cleaning up local tar file..."
rm -f ${IMAGE_NAME}.tar

echo "Step 5/6: Loading image and restarting container on EC2..."
ssh -i $PEM_KEY ${EC2_USER}@${EC2_IP} << EOF
    set -e
    echo "Loading Docker image..."
    sudo docker load -i ${REMOTE_PATH}/${IMAGE_NAME}.tar
    echo "Stopping old container (if exists)..."
    sudo docker stop ${IMAGE_NAME} || true
    sudo docker rm ${IMAGE_NAME} || true
    echo "Starting new container..."
    sudo docker run -d -p ${PORT}:${PORT} --name ${IMAGE_NAME} ${IMAGE_NAME}:latest
    echo "Deployment complete! Running container:"
    sudo docker ps | grep ${IMAGE_NAME}
EOF

echo "Step 6/6: Deployment finished successfully!"


