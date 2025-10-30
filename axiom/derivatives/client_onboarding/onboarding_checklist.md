# Client Onboarding Checklist - Derivatives Platform

## For $5-10M/Year Enterprise Clients (Market Makers)

**Timeline:** 6-8 weeks from contract to production  
**Success Criteria:** <100us Greeks, 99.999% uptime, client profitability

---

## WEEK 1: DISCOVERY & PLANNING

### Client Kickoff Meeting
- [ ] Introduce team (engineering, support, account manager)
- [ ] Review contract and SLAs
- [ ] Understand client architecture
- [ ] Identify integration points
- [ ] Security requirements review
- [ ] Timeline confirmation

### Technical Discovery
- [ ] Current systems inventory
- [ ] Data feeds (OPRA, exchanges, vendors)
- [ ] Execution venues (FIX connections, APIs)
- [ ] Risk systems (internal platforms)
- [ ] Trading strategies overview
- [ ] Volume expectations (req/sec)
- [ ] Latency requirements (confirm <100us)

### Security & Compliance
- [ ] SOC 2 audit report sharing
- [ ] Security questionnaire completed
- [ ] Penetration test results provided
- [ ] Data residency requirements
- [ ] Regulatory compliance (SEC, FINRA, etc.)
- [ ] Business continuity plan reviewed

**Week 1 Deliverable:** Integration plan document, security approved

---

## WEEK 2-3: ENVIRONMENT SETUP

### Infrastructure Provisioning
- [ ] Dedicated namespace in Kubernetes
- [ ] GPU nodes allocated (3-5 pods initially)
- [ ] Database created (PostgreSQL dedicated instance)
- [ ] Redis cluster configured
- [ ] Load balancer setup
- [ ] SSL certificates provisioned
- [ ] VPN/private link established

### API Keys & Authentication
- [ ] Client API keys generated
- [ ] JWT tokens configured
- [ ] Rate limits set (1M+ req/hour for market makers)
- [ ] IP whitelist configured
- [ ] Role-based access control setup

### Monitoring Setup
- [ ] Grafana dashboard created for client
- [ ] Alert routing configured (client ops team)
- [ ] Custom metrics defined (client-specific)
- [ ] Audit logging enabled
- [ ] Performance SLA tracking automated

**Week 2-3 Deliverable:** Test environment accessible, monitoring operational

---

## WEEK 4-5: INTEGRATION & TESTING

### Data Feed Integration
- [ ] OPRA feed connected (if applicable)
- [ ] Exchange APIs tested (CBOE, ISE, PHLX)
- [ ] Market data validation
- [ ] Latency measured (<1ms data feed)
- [ ] Failover tested

### API Integration
- [ ] Python SDK installed on client systems
- [ ] Sample Greeks calculations working
- [ ] Batch processing tested
- [ ] Error handling validated
- [ ] Retry logic confirmed

### Performance Validation
- [ ] Greeks latency: p50, p95, p99 measured
- [ ] Target validation: All <100us
- [ ] Throughput test: >10K req/sec
- [ ] Sustained load test: 1 hour continuous
- [ ] Accuracy verification: 99.99% vs Black-Scholes

### Paper Trading
- [ ] Test environment live trading
- [ ] Full workflow: data → Greeks → hedging → execution
- [ ] P&L tracking operational
- [ ] Risk limits functional
- [ ] Alert system tested
- [ ] 2 weeks of paper trading (minimum)

**Week 4-5 Deliverable:** Paper trading profitable, all systems validated

---

## WEEK 6: PRODUCTION READINESS

### Pre-Production Checklist
- [ ] All tests passed (unit, integration, load)
- [ ] Performance SLAs validated
- [ ] Security audit passed
- [ ] Disaster recovery tested
- [ ] Runbooks reviewed with client ops team
- [ ] Escalation procedures defined
- [ ] Support coverage confirmed (24/7)

### Production Deployment Plan
- [ ] Gradual rollout schedule (10% → 50% → 100%)
- [ ] Rollback procedure documented
- [ ] Success criteria defined
- [ ] Go/no-go checklist prepared
- [ ] Client sign-off obtained

**Week 6 Deliverable:** Production deployment plan approved

---

## WEEK 7: GO-LIVE

### Day 1: 10% Traffic
- [ ] Route 10% of Greeks calculations to Axiom
- [ ] Monitor latency continuously
- [ ] Compare results with existing system
- [ ] Check for any errors
- [ ] Client feedback: All good?

### Day 3: 50% Traffic
- [ ] Increase to 50% if Day 1-2 successful
- [ ] Monitor for 48 hours
- [ ] Performance still <100us?
- [ ] No errors or issues?
- [ ] Client confirms accuracy?

### Day 5: 100% Traffic
- [ ] Full cutover if all successful
- [ ] Monitor closely for 24 hours
- [ ] Performance maintained?
- [ ] Client P&L improving?
- [ ] Declare success!

**Week 7 Deliverable:** Full production deployment successful

---

## WEEK 8: POST-GO-LIVE

### Optimization
- [ ] Review performance data
- [ ] Identify optimization opportunities
- [ ] Fine-tune caching
- [ ] Optimize queries
- [ ] Adjust autoscaling

### Client Success
- [ ] Weekly status calls
- [ ] Performance report (vs SLAs)
- [ ] Feature requests prioritization
- [ ] Training sessions (if needed)
- [ ] Case study preparation (if client approves)

**Week 8 Deliverable:** Stable production, client satisfied, case study started

---

## SUCCESS METRICS

### Technical (Must Achieve)
- [x] Latency p95 < 100us
- [x] Throughput > 10,000 req/sec
- [x] Uptime > 99.9% (target 99.999%)
- [ ] Error rate < 0.01%
- [ ] Accuracy 99.99% vs Black-Scholes

### Business (Client Value)
- [ ] Client P&L improvement > 15%
- [ ] Fill rates improved
- [ ] Hedge costs reduced
- [ ] No operational incidents
- [ ] Client reference provided

### Operational
- [ ] Zero data loss
- [ ] All alerts working
- [ ] Runbooks used successfully
- [ ] Team trained and confident
- [ ] Documentation complete

---

## CLIENT COMMUNICATION

### Weekly Status Email Template

```
Subject: Axiom Derivatives - Week X Status

[Client Name],

Week X Progress:

PERFORMANCE:
✓ Latency: p95 = XXus (target: <100us)
✓ Throughput: XX,XXX req/sec
✓ Uptime: 99.XX%
✓ Accuracy: 99.XX%

MILESTONES:
✓ [Completed items]
→ [In progress]
○ [Upcoming]

METRICS:
- Total Greeks calculated: XXX,XXX
- Average response time: XXus
- Peak throughput: XX,XXX req/sec
- Incidents: X (all resolved)

NEXT WEEK:
- [Key objectives]
- [Expected outcomes]

Please confirm everything looks good.

Best regards,
Axiom Team
```

---

## POST-DEPLOYMENT SUPPORT

### First Month
- [ ] Daily check-ins
- [ ] Performance monitoring
- [ ] Quick bug fixes
- [ ] Feature refinements

### Months 2-3
- [ ] Weekly check-ins
- [ ] Quarterly business review prep
- [ ] Optimization implementations
- [ ] Feature roadmap alignment

### Ongoing
- [ ] Monthly status calls
- [ ] Quarterly business reviews
- [ ] Annual contract renewal discussions
- [ ] Continuous value demonstration

---

**This systematic approach ensures smooth onboarding and long-term client success.**