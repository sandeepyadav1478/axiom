# M&A GitHub Workflows Execution Guide
## How to Trigger and Execute M&A Investment Banking Workflows

## 🎯 M&A Workflow Types

### Automatic CI/CD Workflows (Run on every push/PR)
These run automatically and you see them in the Actions tab:
- ✅ **System Validation & M&A Analysis** - Tests M&A system functionality
- ✅ **Code Quality & Style** - Ensures code standards
- ✅ **Documentation & Setup Validation** - Validates setup guides
- ✅ **Security & Compliance Check** - Security validation

### Manual M&A Operational Workflows (On-Demand Execution)
These are triggered manually for specific M&A deals and operations:
- 📋 **M&A Deal Pipeline Automation** (`ma-deal-pipeline.yml`)
- ⚠️ **M&A Risk Assessment & Management** (`ma-risk-assessment.yml`)
- 💎 **M&A Valuation Model Validation** (`ma-valuation-validation.yml`)
- 📊 **M&A Deal Tracking & Management** (`ma-deal-management.yml`)

## 🚀 How to Execute M&A Workflows

### Method 1: GitHub Web Interface

1. **Navigate to Actions Tab**
   - Go to: `https://github.com/sandeepyadav1478/axiom/actions`

2. **Select M&A Workflow**
   - Click on desired workflow (e.g., "M&A Deal Pipeline Automation")

3. **Click "Run workflow"**
   - Click the "Run workflow" button (blue button on the right)

4. **Fill Parameters**
   ```
   Target company: "DataRobot Inc"
   Deal value estimate: "2800"  (in millions)
   Analysis scope: "comprehensive"
   Priority: "high"
   ```

5. **Execute**
   - Click "Run workflow" to execute

### Method 2: GitHub CLI (Command Line)

```bash
# Install GitHub CLI if needed
brew install gh  # macOS
# or
sudo apt install gh  # Ubuntu

# Login to GitHub
gh auth login

# Execute M&A Deal Pipeline
gh workflow run "M&A Deal Pipeline Automation" \
  -f target_company="DataRobot Inc" \
  -f deal_value_estimate="2800" \
  -f analysis_scope="comprehensive" \
  -f priority="high"

# Execute Risk Assessment
gh workflow run "M&A Risk Assessment & Management" \
  -f target_company="HealthTech AI" \
  -f deal_stage="due_diligence" \
  -f risk_categories="financial,operational,regulatory"

# Execute Valuation Validation
gh workflow run "M&A Valuation Model Validation" \
  -f target_company="CyberSecure Corp" \
  -f valuation_method="comprehensive" \
  -f sensitivity_analysis=true

# Execute Deal Management Dashboard
gh workflow run "M&A Deal Tracking & Management" \
  -f management_action="executive_dashboard" \
  -f deal_filter="active"
```

### Method 3: API Trigger

```bash
# Using curl to trigger workflows via GitHub API
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/sandeepyadav1478/axiom/actions/workflows/ma-deal-pipeline.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "target_company": "DataRobot Inc",
      "deal_value_estimate": "2800",
      "analysis_scope": "comprehensive",
      "priority": "high"
    }
  }'
```

## 📋 Available M&A Workflows & Parameters

### 1. 🏦 M&A Deal Pipeline Automation

**Workflow Name:** `M&A Deal Pipeline Automation`
**File:** `ma-deal-pipeline.yml`

**Parameters:**
- `target_company` (required): Target company name
- `deal_value_estimate` (required): Deal value in millions
- `analysis_scope`: screening | due_diligence | valuation | comprehensive
- `priority`: high | medium | low

**Example Execution:**
```bash
gh workflow run "M&A Deal Pipeline Automation" \
  -f target_company="OpenAI" \
  -f deal_value_estimate="10000" \
  -f analysis_scope="comprehensive" \
  -f priority="high"
```

### 2. ⚠️ M&A Risk Assessment & Management

**Workflow Name:** `M&A Risk Assessment & Management`
**File:** `ma-risk-assessment.yml`

**Parameters:**
- `target_company` (required): Company for risk assessment
- `deal_stage`: screening | due_diligence | valuation | negotiation | closing | integration
- `risk_categories`: Comma-separated list (financial,operational,market,regulatory,integration)

**Example Execution:**
```bash
gh workflow run "M&A Risk Assessment & Management" \
  -f target_company="Tesla Inc" \
  -f deal_stage="due_diligence" \
  -f risk_categories="financial,operational,regulatory"
```

### 3. 💎 M&A Valuation Model Validation

**Workflow Name:** `M&A Valuation Model Validation`
**File:** `ma-valuation-validation.yml`

**Parameters:**
- `target_company` (required): Company for valuation validation
- `valuation_method`: dcf_only | comparables_only | precedent_only | comprehensive
- `sensitivity_analysis`: true | false
- `model_complexity`: basic | detailed | comprehensive

**Example Execution:**
```bash
gh workflow run "M&A Valuation Model Validation" \
  -f target_company="Microsoft" \
  -f valuation_method="comprehensive" \
  -f sensitivity_analysis=true \
  -f model_complexity="detailed"
```

### 4. 📊 M&A Deal Tracking & Management

**Workflow Name:** `M&A Deal Tracking & Management`
**File:** `ma-deal-management.yml`

**Parameters:**
- `management_action`: status_update | deal_review | milestone_tracking | portfolio_analysis | executive_dashboard
- `deal_filter`: all | active | pending_approval | in_negotiation | closing | completed

**Example Execution:**
```bash
gh workflow run "M&A Deal Tracking & Management" \
  -f management_action="executive_dashboard" \
  -f deal_filter="active"
```

## 🕐 When M&A Workflows Trigger Automatically

### Scheduled Triggers
- **Deal Management**: Runs automatically **Monday 8 AM** (weekly executive update) and **Friday 5 PM** (pipeline summary)
- **Valuation Validation**: Runs automatically **9 AM weekdays** for model validation

### Event-Based Triggers
- **Deal Pipeline**: Auto-triggers on PR to branches matching `feature/ma-*` or `deal/*`
- **Risk Assessment**: Auto-triggers on repository dispatch events (`high_value_deal`, `regulatory_concern`)
- **Valuation Validation**: Auto-triggers when valuation files are modified

## 🎯 Why You Don't See Them Running Yet

1. **Manual Trigger Required**: Most M&A workflows are designed for **on-demand execution** for specific deals
2. **No Recent Manual Triggers**: No one has manually triggered them via GitHub UI or CLI yet
3. **Scheduled Times**: Automatic schedules haven't reached execution time yet
4. **Event Conditions**: Event-based triggers haven't been activated yet

## 🚀 Test M&A Workflows Now

To see the M&A workflows in action, execute any of these:

```bash
# Quick test of Deal Pipeline
gh workflow run "M&A Deal Pipeline Automation" \
  -f target_company="Demo Company" \
  -f deal_value_estimate="1000" \
  -f analysis_scope="screening" \
  -f priority="medium"

# Quick test of Risk Assessment  
gh workflow run "M&A Risk Assessment & Management" \
  -f target_company="Demo Target" \
  -f deal_stage="due_diligence"

# Quick test of Executive Dashboard
gh workflow run "M&A Deal Tracking & Management" \
  -f management_action="executive_dashboard"
```

After execution, you'll see them appear in the GitHub Actions tab alongside the existing workflows!

## 🏆 M&A Workflow System Status

✅ **All 4 M&A workflows present** in `.github/workflows/` directory
✅ **Ready for manual execution** via GitHub UI or CLI
✅ **Scheduled execution configured** for regular management reporting  
✅ **Event-based triggers set up** for automatic deal milestone tracking
✅ **Production-ready** for investment banking M&A operations