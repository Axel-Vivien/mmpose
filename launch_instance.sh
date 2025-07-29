#!/usr/bin/env bash
set -e

# Variables
AMI_ID="ami-07ad00b928ed9503b"
KEY_NAME="tanguy"
INSTANCE_TYPE="g6.2xlarge"
REGION="us-east-1"

# Set AWS profile from ~/.aws-profile if it exists
if [[ -f ~/.aws-profile ]]; then
    export AWS_PROFILE=$(cat ~/.aws-profile)
fi

# Display AWS access and configuration
echo "=== AWS Access Information ==="
echo "AWS Profile: ${AWS_PROFILE:-default}"
echo "AWS Identity:"
aws sts get-caller-identity
echo "AWS Region: $REGION"
echo "=============================="

echo "Launching 10 instances in region: $REGION"
echo "Instance type: $INSTANCE_TYPE"
echo "AMI ID: $AMI_ID"
echo "Key pair: $KEY_NAME"

# Use existing IAM role
ROLE_NAME="ec2-s3-access-reboot"

for i in {1..8}; do
    # Format batch number with leading zero
    BATCH_NUM=$(printf "%02d" $i)
    
    echo "Launching instance $i/8 for batch $BATCH_NUM..."
    
    # Create user data script to execute the command at startup
    USER_DATA=$(cat << EOF
#!/bin/bash
cd /home/ec2-user
rm .aws/config
./mmpose/launch_inferencer.sh --json-file s3://veesion-data-reinit-research/axel/all_batches_no_unbiased/video_batch_no_unbiased_${BATCH_NUM}_of_08.json --limit 17500 --shutdown
EOF
)
    echo "$USER_DATA"

    # Launch EC2 instance
    INSTANCE_JSON=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --count 1 \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --iam-instance-profile Name="$ROLE_NAME" \
        --associate-public-ip-address \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":125,"DeleteOnTermination":true}}]' \
        --user-data "$USER_DATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=axel-mmpose-$i}]" \
        --query "Instances[0]" \
        --region "$REGION" \
        --output json)

    INSTANCE_ID=$(echo "$INSTANCE_JSON" | jq -r '.InstanceId')
    echo "Launched EC2 instance $i/10: $INSTANCE_ID (processing batch $BATCH_NUM)"
    
    # Store instance ID for later reference
    echo "$INSTANCE_ID" >> launched_instances.txt
done

echo "Done"