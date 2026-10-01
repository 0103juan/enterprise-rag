# Incident Response Runbook

## Severity levels

SEV1 is a customer-facing outage or any data loss. SEV2 is a degraded service with a workaround. SEV3 is a minor defect with no customer impact. When in doubt, declare the higher severity and downgrade later.

## SEV1 procedure

Page the on-call engineer immediately. An incident commander must be assigned within 10 minutes. Post an update to the public status page every 30 minutes until the incident is resolved. Do not discuss root cause publicly during the incident.

## Postmortems

Every SEV1 and SEV2 incident requires a blameless postmortem, published within five business days. Action items from the postmortem are tracked in Jira under the INC project and reviewed weekly.

## Known error codes

E-4417 means the carrier rate API timed out; fail over to the secondary gateway with `gatewayctl failover carrier`. E-2203 means the label printing queue has stalled; restart the `labeld` service on the affected warehouse node. E-9001 means the tracking database is in read-only mode; this is a SEV1 and must be escalated to the database team.
