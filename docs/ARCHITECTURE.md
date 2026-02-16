# Architecture & Design

The project is structured around the **Abstract Factory** pattern, enabling easy extension to new backends (CPU, GPU, distributed) and numerical methods (Spectral, FEM, etc.).

## Directory Structure

```text
fastHeatSolv/
├── main.py                     # Entry point: parses args, instantiates SimulationFactory
├── config/                     # YAML configuration files for simulation parameters
├── data/
│   └── paths/                  # G-code files for laser paths
├── out/                        # Simulation results (per-run subfolders)
├── core/
│   ├── workflow.py             # Simulation loop (orchestrator, strategy pattern)
│   ├── parameters.py           # Data classes: PhysParams, NumParams, GeomParams, IOParams
│   └── io.py                   # Abstract base class for IOManager
├── interfaces/
│   ├── factory.py              # Abstract Factory: interface for creating solvers, IO managers
│   └── solver.py               # Abstract Product: HeatSolver interface
├── implementations/
│   ├── factories/              # Concrete Factories (CPU, GPU, FEM)
│   ├── solvers/                # Concrete Solvers (Spectral, FEM wrappers)
│   ├── file_io/                # Concrete IO Managers
│   └── physics/                # Low-level physics kernels
└── utils/                      # Utilities (DCT, visualization, logging)
```

## Component Descriptions

- **Client (`main.py`)**: Entry point that parses arguments (CLI/YAML) and injects the appropriate Factory into the Workflow.
- **Workflow (`core/workflow.py`)**: The high-level director that manages the time loop, physics updates, and IO events.
- **Factory Interface (`interfaces/factory.py`)**: Abstract definitions for creating solvers and managers.
- **Solvers (`implementations/solvers/`)**: The mathematical engines.

## Extending the Framework

To add a new solver (e.g., Finite Difference):
1. Create `interfaces/solver.py` compliant implementation in `implementations/solvers/`.
2. Create a factory in `implementations/factories/`.
3. Register the method in `main.py`.

## Architecture Diagram

```mermaid
flowchart TD
    %% --- Styles ---
    classDef default font-family:Arial,color:#333;
    classDef main fill:#2d3436,stroke:#2d3436,color:#fff,rx:5,ry:5;
    classDef note fill:#fff3e0,stroke:#ffb74d,stroke-dasharray: 5 5;
    classDef interface fill:#e3f2fd,stroke:#2196f3,stroke-width:2px,stroke-dasharray: 5 5;
    classDef impl fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef core fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef micro fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef step fill:#f5f6fa,stroke:#7f8fa6,rx:5,ry:5;
    classDef decision fill:#ffebee,stroke:#ef5350,shape:diamond;

    %% --- 1. Entry & Context ---
    subgraph Setup ["🚀 Setup"]
        direction LR
        Main[main.py]:::main
        Context(SimulationContext<br/>Params):::note
        Main --> Context
    end

    %% --- 2. Core Logic ---
    subgraph Core ["⚙️ Core Logic"]
        Workflow[SimulationWorkflow]:::core
        RunLoop((Run Loop)):::core
        
        Main --> Workflow
        Context -.-> Workflow
        Workflow --> RunLoop
    end

    %% --- 3. Architecture (Abstract Factory) ---
    subgraph Architecture ["🏗️ Abstract Factory Architecture"]
        direction TB
        
        %% Layer 1: Interfaces
        subgraph Interfaces ["Abstractions"]
            direction LR
            ISimFactory["SimulationFactory"]:::interface
            IHeat["HeatSolver"]:::interface
            IMicro["MicrostructureSolver"]:::interface
        end

        %% Layer 2: Concrete Factories
        subgraph Factories ["Concrete Factories"]
            direction LR
            CPUFact[CPUSimulationFactory]:::impl
            GPUFact[GPUSimulationFactory]:::impl
        end

        %% Layer 3: Concrete Products
        subgraph Solvers ["Concrete Solvers"]
            direction LR
            CPUSolv[SpectralCPUSolver]:::impl
            GPUSolv[SpectralGPUSolver]:::impl
            TreeSolv[TreeMicroSolver]:::micro
        end

        %% Wiring Architecture
        Workflow -->|Uses| ISimFactory
        RunLoop -->|Calls| IHeat & IMicro

        %% Implementation Links
        CPUFact -.->|Implements| ISimFactory
        GPUFact -.->|Implements| ISimFactory
        
        CPUSolv -.->|Implements| IHeat
        GPUSolv -.->|Implements| IHeat
        TreeSolv -.->|Implements| IMicro

        %% Creation Links
        CPUFact -->|Creates| CPUSolv & TreeSolv
        GPUFact -->|Creates| GPUSolv
    end

    %% --- 4. Process Flow ---
    subgraph Process ["🔄 Simulation Loop (Time Stepping)"]
        direction LR
        Init[1. Init]:::step
        TimeCheck{t < t_end?}:::decision
        
        subgraph Steps ["Step Execution"]
            direction TB
            HeatStep[3a. Heat Solver]:::impl
            MicroStep[3b. Micro Update]:::micro
        end
        
        IOCheck[4. IO Check]:::step
        Log[5. Log]:::step

        Init --> TimeCheck
        TimeCheck -->|Yes| HeatStep
        HeatStep --> MicroStep
        MicroStep --> IOCheck
        IOCheck --> Log
        Log --> TimeCheck
        TimeCheck -->|No| Done((End)):::main
    end

    %% --- Logic Note ---
    MicroDetail["<b>Microstructure Logic:</b>
    1. Get Isotherm (T_melt)
    2. Query AABB Tree
    3. Remove melted seeds
    4. Project & Grow"]:::note
    
    MicroStep -.- MicroDetail
```
