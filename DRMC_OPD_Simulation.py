#!/usr/bin/env python3
"""
================================================================================
DRMC OPD Patient Flow Discrete Event Simulation
================================================================================
A comprehensive simulation model for Davao Regional Medical Center's
Outpatient Department patient flow analysis using SimPy and Streamlit.

Author: Simulation Engineering Team
Purpose: Academic M&S Project - OPD Operations Analysis
================================================================================

INSTALLATION & USAGE:
---------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run the Streamlit application:
   streamlit run DRMC_OPD_Simulation.py

3. Open your browser to the localhost URL shown (typically http://localhost:8501)
================================================================================
"""

import numpy as np
import pandas as pd
import simpy
from simpy import PriorityResource
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import random
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CONFIGURATION AND CONSTANTS
# ============================================================================

# Time conversion: simulation uses minutes, 5 AM = 0 minutes
MINUTES_PER_HOUR = 60
OPD_START_HOUR = 8  # 8:00 AM opening
OPD_END_HOUR = 17   # 5:00 PM closing
PRE_OPENING_START_HOUR = 5  # Pre-opening queue starts at 5:00 AM

# Simulation time boundaries (in minutes from 5:00 AM)
TIME_5AM = 0
TIME_7AM = 2 * 60    # 120
TIME_8AM = 3 * 60    # 180 (OPD opens)
TIME_10AM = 5 * 60   # 300
TIME_1PM = 8 * 60    # 480
TIME_5PM = 12 * 60   # 720

# Priority Levels (lower number = higher priority)
PRIORITY_GENERAL = 2
PRIORITY_EXPRESS = 1  # Senior Citizens, PWDs, Pregnant, Infants

# Arrival Phase Definitions (in minutes from 5 AM)
ARRIVAL_PHASES = {
    'dawn_surge': {'start': TIME_5AM, 'end': TIME_7AM, 'label': 'Dawn Surge (5-7 AM)'},
    'peak_congestion': {'start': TIME_7AM, 'end': TIME_10AM, 'label': 'Peak Congestion (7-10 AM)'},
    'midday': {'start': TIME_10AM, 'end': TIME_1PM, 'label': 'Midday (10 AM-1 PM)'},
    'afternoon': {'start': TIME_1PM, 'end': TIME_5PM, 'label': 'Afternoon (1-5 PM)'}
}

# Patient Routing Distribution
CLINIC_DISTRIBUTION = {
    'Internal Medicine': 0.27,    # 25-30%
    'Pediatrics': 0.18,          # 15-20%
    'OB-GYN': 0.16,              # 15-18%
    'Surgery': 0.11,              # 10-12%
    'Orthopedics': 0.07,          # 6-8%
    'Ophthalmology': 0.05,        # 4-6%
    'Pediatric Immunization': 0.04,  # 3-5%
    'Dental': 0.03,               # 2-4%
    'Dermatology': 0.025,         # 2-3%
    'Unassigned': 0.065            # 5-7%
}

# Patient Source Distribution
PATIENT_SOURCE_DIST = {
    'Scheduled': 0.25,
    'Follow-up': 0.35,
    'Walk-in': 0.20,
    'Referred': 0.20
}

# Post-Consultation Routing
POST_CONSULT_DIST = {
    'Direct Exit': 0.45,
    'Diagnostics': 0.25,
    'Follow-up Scheduling': 0.20,
    'Referral/Admission': 0.10
}

# ============================================================================
# SECTION 2: DATA CLASSES AND STRUCTURES
# ============================================================================

@dataclass
class Patient:
    """Represents a patient entity in the simulation."""
    patient_id: int
    patient_type: str  # 'Scheduled', 'Follow-up', 'Walk-in', 'Referred'
    priority: int  # 1 = Express, 2 = General
    is_new: bool  # True for new patients
    clinic: str
    arrival_time: float  # Simulation time in minutes from 5 AM
    demographics: str  # 'Adult', 'Senior', 'PWD', 'Pregnant', 'Infant'
    
    # Timestamps for various stages
    gate_entry_time: Optional[float] = None
    vitals_time: Optional[float] = None
    registration_complete_time: Optional[float] = None
    clinic_checkin_time: Optional[float] = None
    consultation_start_time: Optional[float] = None
    consultation_end_time: Optional[float] = None
    exit_time: Optional[float] = None
    
    # Diagnostic flags
    requires_diagnostics: bool = False
    requires_followup: bool = False
    requires_referral: bool = False

@dataclass
class KPICollector:
    """Collects and stores all KPI metrics during simulation."""
    # Queue lengths over time
    pre_opening_queue: List[Tuple[float, int]] = field(default_factory=list)
    gate_queue_history: List[Tuple[float, int]] = field(default_factory=list)
    registration_queue_history: List[Tuple[float, int]] = field(default_factory=list)
    clinic_queue_history: List[Tuple[float, int]] = field(default_factory=list)
    
    # Waiting times
    gate_wait_times: List[float] = field(default_factory=list)
    gate_verify_vitals_times: List[float] = field(default_factory=list)
    vitals_wait_times: List[float] = field(default_factory=list)
    registration_wait_times: List[float] = field(default_factory=list)
    clinic_wait_times: List[float] = field(default_factory=list)
    consultation_wait_times: List[float] = field(default_factory=list)
    
    # Service times
    gate_service_times: List[float] = field(default_factory=list)
    registration_service_times: List[float] = field(default_factory=list)
    consultation_service_times: List[float] = field(default_factory=list)
    
    # Doctor staffing interruption events
    doctor_interruption_events: List[Dict] = field(default_factory=list)
    
    # Total length of stay
    length_of_stay: List[float] = field(default_factory=list)
    
    # Throughput
    patients_completed: int = 0
    patients_arrived: int = 0
    
    # Resource utilization tracking
    resource_busy_time: Dict[str, float] = field(default_factory=dict)
    resource_total_time: Dict[str, float] = field(default_factory=dict)
    
    # Detailed patient records
    patient_records: List[Dict] = field(default_factory=list)

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    # Volume and timing
    daily_volume: int = 750
    num_replications: int = 3
    simulation_day: int = 1  # For scenario D
    
    # Staff counts
    gate_security_count: int = 2
    verification_staff_count: int = 2
    health_aide_count: int = 2
    registration_staff_count: int = 3
    clinic_routing_staff_count: int = 2
    sub_clinic_nursing_count: int = 4
    
    # Physician staffing
    morning_physician_count: int = 38
    afternoon_physician_count: int = 28
    physician_session_cap: int = 18
    
    # Arrival rates by phase (patients per hour)
    dawn_surge_rate: float = 40
    peak_rate_mon: float = 110  # Monday peak
    peak_rate_normal: float = 70
    midday_rate: float = 38
    afternoon_rate: float = 20
    
    # Doctor staffing interruption parameters
    doctor_interruption_frequency: float = 1.5  # per day
    doctor_interruption_min_duration: int = 180
    doctor_interruption_mode_duration: int = 240
    doctor_interruption_max_duration: int = 300
    
    # Priority patient percentage
    priority_patient_percentage: float = 0.25  # 25%
    express_patient_percentage: float = 0.15   # 15% of priority
    
    # New patient percentage
    new_patient_percentage: float = 0.30  # 30%
    
    # No-show rate (Scenario C)
    no_show_rate: float = 0.15  # 15%
    
    # Scenario flags
    scenario_a_enabled: bool = False  # Front-end scaling
    scenario_b_enabled: bool = False  # Consultation caps
    scenario_c_enabled: bool = False  # No-show optimization
    scenario_d_enabled: bool = False  # Mid-week staffing

# ============================================================================
# SECTION 3: SIMULATION MODEL
# ============================================================================

class DRMC_OPD_Simulation:
    """
    Discrete Event Simulation model for DRMC Outpatient Department.
    
    Models patient flow from pre-opening arrival through gate screening,
    verification/vitals, registration, sub-clinic check-in, consultation,
    and post-consultation routing.
    """
    
    def __init__(self, env: simpy.Environment, config: SimulationConfig, 
                 kpi: KPICollector):
        self.env = env
        self.config = config
        self.kpi = kpi
        
        # Initialize resources
        self._setup_resources()
        
        # Patient counter
        self.patient_counter = 0
        
        # Track clinic queues for doctor staffing interruptions
        self.clinic_physicians_active = {clinic: True for clinic in CLINIC_DISTRIBUTION.keys()}
        
        # Live counter for pre-opening queue
        self.active_pre_opening = 0
        
    def _setup_resources(self):
        """Initialize all SimPy resources with configured capacities."""
        
        # Apply scenario modifications
        if self.config.scenario_a_enabled:
            # Front-end scaling: 3-4 verification, 5-6 registration
            verification_count = max(3, self.config.verification_staff_count)
            registration_count = max(5, self.config.registration_staff_count)
        else:
            verification_count = self.config.verification_staff_count
            registration_count = self.config.registration_staff_count
        
        # Apply scenario D (mid-week staffing reduction)
        if self.config.scenario_d_enabled and self.config.simulation_day in [3, 4]:  # Wed/Thu
            reduction_factor = 0.75
            verification_count = max(1, int(verification_count * reduction_factor))
            registration_count = max(2, int(registration_count * reduction_factor))
        
        # Gate Security (outer perimeter)
        self.gate_security = simpy.PriorityResource(
            self.env, capacity=self.config.gate_security_count
        )
        
        # Verification Staff (appointment code/referral verification + vitals)
        self.verification_staff = simpy.PriorityResource(
            self.env, capacity=verification_count
        )
        
        # Health Aides (vitals capture)
        self.health_aides = simpy.PriorityResource(
            self.env, capacity=self.config.health_aide_count
        )
        
        # Registration Staff (central registration encoders)
        self.registration_staff = simpy.PriorityResource(
            self.env, capacity=registration_count
        )
        
        # Clinic Routing Staff
        self.clinic_routing = simpy.PriorityResource(
            self.env, capacity=self.config.clinic_routing_staff_count
        )
        
        # Sub-Clinic Nursing Staff (manages check-in boxes and priority rotation)
        self.sub_clinic_nursing = {
            clinic: simpy.PriorityResource(self.env, capacity=max(1, self.config.sub_clinic_nursing_count // 4))
            for clinic in CLINIC_DISTRIBUTION.keys()
        }
        
        # OPD Physicians (allocated by clinic)
        # Distribute physicians based on clinic demand
        total_physicians = self.config.morning_physician_count
        
        # For simplicity, allocate based on clinic proportion
        self.physician_pools = {}
        for clinic, proportion in CLINIC_DISTRIBUTION.items():
            # Morning session physicians
            morning_docs = max(2, int(total_physicians * proportion))
            self.physician_pools[clinic] = {
                'morning': simpy.PriorityResource(self.env, capacity=morning_docs),
                'afternoon': simpy.PriorityResource(self.env, capacity=max(1, int(self.config.afternoon_physician_count * proportion))),
                'patients_seen': {'morning': 0, 'afternoon': 0},
                'session_cap': self.config.physician_session_cap if self.config.scenario_b_enabled else 100
            }
        
        # Diagnostic Channels
        self.diagnostic_channels = simpy.PriorityResource(self.env, capacity=4)
        
        # Pharmacy Counters
        self.pharmacy_counters = simpy.PriorityResource(self.env, capacity=3)
        
        # Follow-up Scheduling Counter
        self.followup_desk = simpy.PriorityResource(self.env, capacity=2)
        
        # Initialize resource tracking
        self._init_resource_tracking()
    
    def _init_resource_tracking(self):
        """Initialize resource utilization tracking."""
        resources = [
            'gate_security', 'verification_staff', 'health_aides',
            'registration_staff', 'clinic_routing', 'sub_clinic_nursing', 'physicians'
        ]
        for r in resources:
            if r not in self.kpi.resource_total_time:
                self.kpi.resource_total_time[r] = 0.0
            if r not in self.kpi.resource_busy_time:
                self.kpi.resource_busy_time[r] = 0.0
    
    def triangular_time(self, a: float, b: float, c: float) -> float:
        """Generate random time using triangular distribution (in minutes)."""
        return stats.triang.rvs(loc=a, scale=c-a, c=(b-a)/(c-a))
    
    def poisson_arrival_rate(self, phase: str, is_monday: bool = False) -> float:
        """
        Calculate arrival rate based on phase and day.
        Returns patients per hour.
        """
        rates = {
            'dawn_surge': self.config.dawn_surge_rate,
            'peak_congestion': self.config.peak_rate_mon if is_monday else self.config.peak_rate_normal,
            'midday': self.config.midday_rate,
            'afternoon': self.config.afternoon_rate
        }
        return rates.get(phase, self.config.midday_rate)
    
    def get_arrival_phase(self, time: float) -> str:
        """Determine arrival phase based on simulation time."""
        if time < TIME_7AM:
            return 'dawn_surge'
        elif time < TIME_10AM:
            return 'peak_congestion'
        elif time < TIME_1PM:
            return 'midday'
        else:
            return 'afternoon'
    
    def calculate_inter_arrival_time(self, time: float, is_monday: bool = False) -> float:
        """
        Calculate inter-arrival time based on current phase.
        Returns time in minutes.
        """
        phase = self.get_arrival_phase(time)
        rate_per_hour = self.poisson_arrival_rate(phase, is_monday)
        
        # Convert to per-minute rate and calculate inter-arrival time
        rate_per_minute = rate_per_hour / 60.0
        
        # Handle zero rate case
        if rate_per_minute <= 0:
            return 60.0  # Default to 1 hour
        
        return np.random.exponential(1.0 / rate_per_minute)
    
    def generate_patient(self) -> Patient:
        """Generate a new patient with appropriate attributes."""
        self.patient_counter += 1
        
        # Determine patient source (Scheduled, Follow-up, Walk-in, Referred)
        sources = list(PATIENT_SOURCE_DIST.keys())
        source_probs = list(PATIENT_SOURCE_DIST.values())
        patient_type = np.random.choice(sources, p=source_probs)
        
        # Determine priority (Express or General)
        is_express = np.random.random() < self.config.express_patient_percentage
        priority = PRIORITY_EXPRESS if is_express else PRIORITY_GENERAL
        
        # Demographics for priority determination
        demo_rand = np.random.random()
        if demo_rand < 0.02:
            demographics = 'Infant'
            priority = PRIORITY_EXPRESS
        elif demo_rand < 0.08:
            demographics = 'Pregnant'
            priority = PRIORITY_EXPRESS
        elif demo_rand < 0.15:
            demographics = 'Senior'
            priority = PRIORITY_EXPRESS
        elif demo_rand < 0.22:
            demographics = 'PWD'
            priority = PRIORITY_EXPRESS
        else:
            demographics = 'Adult'
        
        # Determine if new patient
        is_new = np.random.random() < self.config.new_patient_percentage
        
        # Assign clinic based on distribution
        clinics = list(CLINIC_DISTRIBUTION.keys())
        clinic_probs = list(CLINIC_DISTRIBUTION.values())
        clinic = np.random.choice(clinics, p=clinic_probs)
        
        return Patient(
            patient_id=self.patient_counter,
            patient_type=patient_type,
            priority=priority,
            is_new=is_new,
            clinic=clinic,
            arrival_time=self.env.now,
            demographics=demographics
        )
    
    def patient_arrival_process(self, is_monday: bool = False):
        """Generate patient arrivals using Poisson process."""
        patients_generated = 0
        target_patients = self.config.daily_volume
        
        # Adjust for no-show rate (Scenario C)
        if self.config.scenario_c_enabled:
            # Reduce effective arrivals by no-show rate
            effective_arrivals = int(target_patients * (1 - self.config.no_show_rate))
        else:
            effective_arrivals = target_patients
        
        while patients_generated < effective_arrivals:
            # Calculate inter-arrival time
            inter_arrival = self.calculate_inter_arrival_time(self.env.now, is_monday)
            
            # Don't generate arrivals after 5 PM (720 minutes)
            if self.env.now + inter_arrival > TIME_5PM + 60:  # Allow 1 hour buffer
                break
            
            yield self.env.timeout(inter_arrival)
            
            # Generate patient
            patient = self.generate_patient()
            if self.env.now < TIME_8AM:
                self.active_pre_opening += 1
            self.kpi.patients_arrived += 1
            patients_generated += 1
            
            # Start patient journey
            self.env.process(self.patient_journey(patient))
    
    def patient_journey(self, patient: Patient):
        """
        Main patient flow process through all stages.
        Implements the complete patient journey with all checkpoints.
        """
        patient_record = {
            'patient_id': patient.patient_id,
            'patient_type': patient.patient_type,
            'priority': patient.priority,
            'clinic': patient.clinic,
            'demographics': patient.demographics,
            'is_new': patient.is_new,
            'arrival_time': patient.arrival_time
        }
        
        try:
            # ========== STAGE 1: Gate Entry ==========
            with self.gate_security.request(priority=patient.priority) as req:
                yield req
                gate_start = self.env.now
                gate_service = self.triangular_time(0.3, 1.0, 3.0)  # Gate processing
                yield self.env.timeout(gate_service)
                patient.gate_entry_time = self.env.now
                gate_wait = patient.gate_entry_time - patient.arrival_time
                
                self.kpi.gate_wait_times.append(gate_wait)
                self.kpi.gate_service_times.append(gate_service)
                self.kpi.resource_busy_time['gate_security'] += gate_service
            
            # ========== STAGE 2: Verification & Vitals ==========
            # Verification (appointment code / referral slip)
            with self.verification_staff.request(priority=patient.priority) as req:
                yield req
                verify_start = self.env.now
                verification_time = self.triangular_time(0.5, 2.5, 15.0)  # From proposal
                yield self.env.timeout(verification_time)
                patient.vitals_time = self.env.now
                self.kpi.resource_busy_time['verification_staff'] += verification_time
                
            # Vitals capture
            with self.health_aides.request(priority=patient.priority) as req:
                yield req
                vitals_time = self.triangular_time(1.0, 3.0, 8.0)
                yield self.env.timeout(vitals_time)
                self.kpi.resource_busy_time['health_aides'] += vitals_time
            
            # After both verification and vitals are done:
            verify_vitals_end = self.env.now
            gate_verify_vitals_time = verify_vitals_end - patient.arrival_time
            self.kpi.gate_verify_vitals_times.append(gate_verify_vitals_time)
            
            vitals_wait = patient.vitals_time - gate_start - verification_time
            self.kpi.vitals_wait_times.append(vitals_wait)
            
            # ========== STAGE 3: Central Registration ==========
            with self.registration_staff.request(priority=patient.priority) as req:
                yield req
                reg_start = self.env.now
                
                if patient.is_new:
                    # New patient registration (longer process)
                    registration_time = self.triangular_time(7, 17.5, 60)
                else:
                    # Returning patient (faster)
                    registration_time = self.triangular_time(1, 4, 20)
                
                yield self.env.timeout(registration_time)
                patient.registration_complete_time = self.env.now
                
                reg_wait = patient.registration_complete_time - patient.vitals_time
                reg_service = registration_time
                
                self.kpi.registration_wait_times.append(reg_wait)
                self.kpi.registration_service_times.append(reg_service)
                self.kpi.resource_busy_time['registration_staff'] += registration_time
            
            # ========== STAGE 4: Sub-Clinic Check-in ==========
            with self.clinic_routing.request(priority=patient.priority) as req:
                yield req
                routing_time = self.triangular_time(0.5, 2.0, 5.0)
                yield self.env.timeout(routing_time)
                self.kpi.resource_busy_time['clinic_routing'] += routing_time
            
            # Sub-clinic check-in (nursing staff)
            clinic_nursing = self.sub_clinic_nursing.get(patient.clinic, self.sub_clinic_nursing['Internal Medicine'])
            with clinic_nursing.request(priority=patient.priority) as req:
                yield req
                checkin_time = self.triangular_time(0.5, 1.5, 4.0)
                yield self.env.timeout(checkin_time)
                patient.clinic_checkin_time = self.env.now
                self.kpi.resource_busy_time['sub_clinic_nursing'] += checkin_time
            
            # ========== STAGE 5: Doctor Consultation ==========
            clinic_pool = self.physician_pools.get(patient.clinic, self.physician_pools['Internal Medicine'])
            
            # Determine session (morning or afternoon)
            time_since_5am = self.env.now
            if time_since_5am < TIME_8AM + 240:  # Before 12 PM
                session = 'morning'
            else:
                session = 'afternoon'
            
            # Check physician capacity
            if clinic_pool['patients_seen'][session] >= clinic_pool['session_cap']:
                # Patient redirected or rescheduled (wait for next session or exit)
                yield self.env.timeout(30)  # Brief wait
                if clinic_pool['patients_seen'][session] >= clinic_pool['session_cap']:
                    # Exit due to capacity
                    patient.exit_time = self.env.now
                    patient_record['exit_reason'] = 'session_cap_reached'
                    patient_record['length_of_stay'] = patient.exit_time - patient.arrival_time
                    self.kpi.patient_records.append(patient_record)
                    return
            
            # Request physician (priority-based queuing with 1:3 rotation)
            # For priority patients, insert them ahead of every 3 general patients
            physician_req = self.physician_pools[patient.clinic][session if session == 'morning' else 'afternoon'].request(
                priority=patient.priority
            )
            
            clinic_queue_start = self.env.now
            with physician_req as req:
                yield req
                clinic_queue_wait = self.env.now - clinic_queue_start
                self.kpi.clinic_wait_times.append(clinic_queue_wait)
                
                patient.consultation_start_time = self.env.now
                
                # Consultation service time (triangular distribution from proposal)
                consultation_time = self.triangular_time(5, 15, 45)
                yield self.env.timeout(consultation_time)
                
                patient.consultation_end_time = self.env.now
                clinic_pool['patients_seen'][session] += 1
                
                cons_wait = patient.consultation_start_time - patient.clinic_checkin_time
                cons_service = consultation_time
                
                self.kpi.consultation_wait_times.append(cons_wait)
                self.kpi.consultation_service_times.append(cons_service)
                self.kpi.resource_busy_time['physicians'] += consultation_time
            
            # ========== STAGE 6: Post-Consultation Routing ==========
            # Determine post-consultation path
            paths = list(POST_CONSULT_DIST.keys())
            path_probs = list(POST_CONSULT_DIST.values())
            
            # Check if diagnostics required based on clinic (40% aggregate demand)
            if np.random.random() < 0.40:
                patient.requires_diagnostics = True
            
            # Determine primary routing
            routing = np.random.choice(paths, p=path_probs)
            
            # Handle each routing path
            if routing == 'Direct Exit' or not patient.requires_diagnostics:
                # Simple exit with prescription/instructions
                yield self.env.timeout(self.triangular_time(2, 5, 15))  # Pharmacy wait if needed
                patient.exit_time = self.env.now
                
            elif routing == 'Diagnostics':
                # 25% go to diagnostics
                with self.diagnostic_channels.request(priority=patient.priority) as req:
                    yield req
                    diagnostic_time = self.triangular_time(15, 30, 60)
                    yield self.env.timeout(diagnostic_time)
                
                # May also need pharmacy
                if np.random.random() < 0.6:
                    with self.pharmacy_counters.request(priority=patient.priority) as req:
                        yield req
                        pharmacy_time = self.triangular_time(5, 15, 40)
                        yield self.env.timeout(pharmacy_time)
                
                patient.exit_time = self.env.now
                
            elif routing == 'Follow-up Scheduling':
                # 20% schedule follow-up
                with self.followup_desk.request(priority=patient.priority) as req:
                    yield req
                    followup_time = self.triangular_time(3, 8, 20)
                    yield self.env.timeout(followup_time)
                
                # May need pharmacy
                if np.random.random() < 0.5:
                    with self.pharmacy_counters.request(priority=patient.priority) as req:
                        yield req
                        pharmacy_time = self.triangular_time(5, 15, 40)
                        yield self.env.timeout(pharmacy_time)
                
                patient.exit_time = self.env.now
                
            else:  # Referral/Admission
                # 10% referred or admitted
                with self.followup_desk.request(priority=patient.priority) as req:
                    yield req
                    referral_time = self.triangular_time(5, 15, 45)
                    yield self.env.timeout(referral_time)
                
                patient.exit_time = self.env.now
            
            # ========== EXIT ==========
            self.kpi.patients_completed += 1
            los = patient.exit_time - patient.arrival_time
            self.kpi.length_of_stay.append(los)
            
            patient_record['gate_wait'] = self.kpi.gate_wait_times[-1] if self.kpi.gate_wait_times else 0
            patient_record['gate_verify_vitals'] = self.kpi.gate_verify_vitals_times[-1] if self.kpi.gate_verify_vitals_times else 0
            patient_record['registration_wait'] = self.kpi.registration_wait_times[-1] if self.kpi.registration_wait_times else 0
            patient_record['clinic_wait'] = self.kpi.clinic_wait_times[-1] if self.kpi.clinic_wait_times else 0
            patient_record['consultation_wait'] = cons_wait
            patient_record['consultation_service'] = cons_service
            patient_record['length_of_stay'] = los
            patient_record['exit_reason'] = 'completed'
            
            self.kpi.patient_records.append(patient_record)
            
        except Exception as e:
            # Handle any interruptions (e.g., doctor freeze)
            patient.exit_time = self.env.now
            patient_record['exit_reason'] = 'interrupted'
            patient_record['error'] = str(e)
            self.kpi.patient_records.append(patient_record)
        finally:
            if patient.arrival_time < TIME_8AM:
                self.active_pre_opening = max(0, self.active_pre_opening - 1)
    
    def doctor_interruption_process(self):
        """
        Simulate doctor staffing interruption events where physicians are pulled to ER/ward duties.
        Triggers 1-2 times per day per clinic, lasting 3-5 hours.
        Models these calls as a partial capacity reduction where doctor resources are
        temporarily reduced by approximately 25% (meaning 75% of staffing remains active).
        """
        while True:
            # Wait for random interval until next freeze event
            # Average of 1.5 per day, so ~480 minutes between events on average
            time_to_interruption = np.random.exponential(480 / self.config.doctor_interruption_frequency)
            yield self.env.timeout(time_to_interruption)
            
            # Only trigger during operational hours
            if self.env.now < TIME_8AM or self.env.now > TIME_5PM:
                continue
            
            # Pick a random clinic to interrupt
            clinics = list(self.physician_pools.keys())
            interrupted_clinic = np.random.choice(clinics)
            
            interruption_start = self.env.now
            
            # Interruption duration: triangular(180, 240, 300) minutes = 3-5 hours
            interruption_duration = self.triangular_time(
                self.config.doctor_interruption_min_duration,
                self.config.doctor_interruption_mode_duration,
                self.config.doctor_interruption_max_duration
            )
            
            # Record interruption event
            self.kpi.doctor_interruption_events.append({
                'clinic': interrupted_clinic,
                'start_time': interruption_start,
                'duration': interruption_duration,
                'start_hour': 5 + interruption_start / 60  # Convert to hour from 5 AM
            })
            
            # Mark clinic as partially active (for wait time impact/KPI purposes)
            self.clinic_physicians_active[interrupted_clinic] = False
            
            # Apply 25% capacity reduction (leaving ~75% active)
            pool = self.physician_pools[interrupted_clinic]
            normal_morning_cap = pool['morning'].capacity
            normal_afternoon_cap = pool['afternoon'].capacity
            
            reduced_morning_cap = max(1, int(round(normal_morning_cap * 0.75)))
            reduced_afternoon_cap = max(1, int(round(normal_afternoon_cap * 0.75)))
            
            pool['morning']._capacity = reduced_morning_cap
            pool['afternoon']._capacity = reduced_afternoon_cap
            
            yield self.env.timeout(interruption_duration)
            
            # Restore normal capacities and trigger queue processing
            pool['morning']._capacity = normal_morning_cap
            pool['afternoon']._capacity = normal_afternoon_cap
            
            # Restore clinic availability
            self.clinic_physicians_active[interrupted_clinic] = True
    
    def queue_monitoring_process(self):
        """
        Background process to monitor and record queue lengths over time.
        Samples queue states at regular intervals.
        """
        while True:
            # Record current queue lengths
            current_time = self.env.now
            
            # Pre-opening queue (before 8 AM)
            if current_time < TIME_8AM:
                self.kpi.pre_opening_queue.append((current_time, self.active_pre_opening))
            
            # Gate queue
            gate_queue = len(self.gate_security.queue)
            self.kpi.gate_queue_history.append((current_time, gate_queue))
            
            # Registration queue
            reg_queue = len(self.registration_staff.queue)
            self.kpi.registration_queue_history.append((current_time, reg_queue))
            
            # Clinic queue
            clinic_queue = sum(len(pool['morning'].queue) for pool in self.physician_pools.values())
            self.kpi.clinic_queue_history.append((current_time, clinic_queue))
            
            # Track capacity-minutes dynamically
            self.kpi.resource_total_time['gate_security'] += self.gate_security.capacity
            self.kpi.resource_total_time['verification_staff'] += self.verification_staff.capacity
            self.kpi.resource_total_time['health_aides'] += self.health_aides.capacity
            self.kpi.resource_total_time['registration_staff'] += self.registration_staff.capacity
            self.kpi.resource_total_time['clinic_routing'] += self.clinic_routing.capacity
            self.kpi.resource_total_time['sub_clinic_nursing'] += sum(r.capacity for r in self.sub_clinic_nursing.values())
            
            if current_time < TIME_8AM + 240:
                self.kpi.resource_total_time['physicians'] += sum(pool['morning'].capacity for pool in self.physician_pools.values())
            else:
                self.kpi.resource_total_time['physicians'] += sum(pool['afternoon'].capacity for pool in self.physician_pools.values())
            
            # Sample every minute
            yield self.env.timeout(1)


# ============================================================================
# SECTION 4: SIMULATION ENGINE
# ============================================================================

def run_simulation(config: SimulationConfig, progress_callback=None) -> KPICollector:
    """
    Run the OPD simulation with the given configuration.
    
    Args:
        config: SimulationConfig object with all parameters
        progress_callback: Optional callback for progress updates
        
    Returns:
        KPICollector with all collected metrics
    """
    # Initialize KPI collector
    kpi = KPICollector()
    
    # Run multiple replications for statistical validity
    all_results = []
    
    for replication in range(config.num_replications):
        # Create new environment for each replication
        env = simpy.Environment()
        
        # Create simulation instance
        sim = DRMC_OPD_Simulation(env, config, kpi)
        
        # Start background processes
        env.process(sim.queue_monitoring_process())
        
        # Start doctor staffing interruption process (only if enabled or baseline)
        if config.scenario_b_enabled or config.num_replications > 0:
            env.process(sim.doctor_interruption_process())
        
        # Determine if this is Monday (for higher arrival rates)
        is_monday = (config.simulation_day % 7 == 1)  # Day 1 = Monday
        
        # Start patient arrival process
        env.process(sim.patient_arrival_process(is_monday=is_monday))
        
        # Run simulation for full day (5 AM to 6 PM = 13 hours = 780 minutes)
        env.run(until=780)
        
        if progress_callback:
            progress_callback((replication + 1) / config.num_replications)
        
        all_results.append(kpi)
    
    return kpi


def safe_mean(lst):
    """Safe mean that handles empty lists."""
    return np.mean(lst) if lst else 0.0

def safe_std(lst):
    """Safe std that handles empty or single-element lists."""
    if not lst or len(lst) < 2:
        return 0.0
    return np.std(lst)

def safe_percentile(lst, q):
    """Safe percentile that handles empty lists."""
    if not lst:
        return 0.0
    return np.percentile(lst, q)

def aggregate_results(results_list: List[KPICollector]) -> Dict[str, Any]:
    """
    Aggregate results from multiple simulation replications.
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not results_list:
        return {}
    
    # Helper to get means across replications
    def get_replication_means(field_name):
        means = []
        for r in results_list:
            val = getattr(r, field_name, [])
            if val:
                means.append(np.mean(val))
        return means
    
    aggregated = {
        'gate_wait_mean': safe_mean(get_replication_means('gate_wait_times')),
        'gate_wait_std': safe_std(get_replication_means('gate_wait_times')),
        'gate_wait_p95': np.median([safe_percentile(r.gate_wait_times, 95) for r in results_list]),
        
        'gate_verify_vitals_mean': safe_mean(get_replication_means('gate_verify_vitals_times')),
        'gate_verify_vitals_std': safe_std(get_replication_means('gate_verify_vitals_times')),
        'gate_verify_vitals_p95': np.median([safe_percentile(r.gate_verify_vitals_times, 95) for r in results_list]),
        
        'registration_wait_mean': safe_mean(get_replication_means('registration_wait_times')),
        'registration_wait_std': safe_std(get_replication_means('registration_wait_times')),
        'registration_wait_p95': np.median([safe_percentile(r.registration_wait_times, 95) for r in results_list]),
        
        'clinic_wait_mean': safe_mean(get_replication_means('clinic_wait_times')),
        'clinic_wait_std': safe_std(get_replication_means('clinic_wait_times')),
        'clinic_wait_p95': np.median([safe_percentile(r.clinic_wait_times, 95) for r in results_list]),
        
        'consultation_wait_mean': safe_mean(get_replication_means('consultation_wait_times')),
        'consultation_service_mean': safe_mean(get_replication_means('consultation_service_times')),
        
        'los_mean': safe_mean(get_replication_means('length_of_stay')),
        'los_std': safe_std(get_replication_means('length_of_stay')),
        'los_p95': np.median([safe_percentile(r.length_of_stay, 95) for r in results_list]),
        
        'patients_completed_total': sum(r.patients_completed for r in results_list) / len(results_list),
        'patients_arrived_total': sum(r.patients_arrived for r in results_list) / len(results_list),
        'throughput_rate': np.mean([r.patients_completed / max(1, r.patients_arrived) for r in results_list]),
        
        'interruption_events_count': safe_mean([len(r.doctor_interruption_events) for r in results_list]),
        'interruption_total_duration': safe_mean([sum(e['duration'] for e in r.doctor_interruption_events) for r in results_list]),
        
        'gate_queue_max': max([max([q for t, q in r.gate_queue_history], default=0) for r in results_list]),
        'registration_queue_max': max([max([q for t, q in r.registration_queue_history], default=0) for r in results_list]),
        'clinic_queue_max': max([max([q for t, q in r.clinic_queue_history], default=0) for r in results_list]),
    }
    
    return aggregated


# ============================================================================
# SECTION 5: STREAMLIT UI AND DASHBOARD
# ============================================================================

def create_sidebar_config():
    """Create and return simulation configuration from sidebar controls."""
    
    st.sidebar.header("Simulation Configuration")
    
    # Basic Parameters
    st.sidebar.subheader("Basic Parameters")
    config = SimulationConfig()
    
    config.daily_volume = st.sidebar.slider(
        "Daily Patient Volume",
        min_value=500, max_value=1000, value=750, step=50,
        help="Target number of patients per day (500-1000)"
    )
    
    config.num_replications = st.sidebar.slider(
        "Simulation Replications",
        min_value=1, max_value=10, value=3,
        help="Number of independent simulation runs for statistical averaging"
    )
    
    config.simulation_day = st.sidebar.selectbox(
        "Simulation Day of Week",
        options=list(range(1, 6)), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][x-1],
        help="Day affects arrival patterns (Monday has higher surge)"
    )
    
    st.sidebar.divider()
    
    # Staffing Parameters
    st.sidebar.subheader("Staffing Configuration")
    
    st.sidebar.markdown("**Front-End Staff**")
    config.gate_security_count = st.sidebar.slider(
        "Gate Security Staff", 1, 5, 2,
        help="Staff at outer perimeter for crowd control"
    )
    config.verification_staff_count = st.sidebar.slider(
        "Verification Staff (Baseline)", 1, 4, 2,
        help="Staff for appointment/referral verification"
    )
    config.health_aide_count = st.sidebar.slider(
        "Health Aides", 1, 5, 2,
        help="Staff for vital signs capture"
    )
    config.registration_staff_count = st.sidebar.slider(
        "Registration Staff (Baseline)", 2, 6, 3,
        help="Central registration encoders"
    )
    
    st.sidebar.markdown("**Clinical Staff**")
    config.clinic_routing_staff_count = st.sidebar.slider(
        "Clinic Routing Staff", 1, 4, 2,
        help="Staff routing patients to appropriate clinics"
    )
    config.sub_clinic_nursing_count = st.sidebar.slider(
        "Sub-Clinic Nursing Staff", 2, 8, 4,
        help="Nursing staff managing clinic check-in boxes"
    )
    
    st.sidebar.markdown("**Physician Staffing**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        config.morning_physician_count = st.slider(
            "Morning Physicians", 20, 50, 38,
            help="Number of doctors in morning session"
        )
    with col2:
        config.afternoon_physician_count = st.slider(
            "Afternoon Physicians", 15, 40, 28,
            help="Number of doctors in afternoon session"
        )
    config.physician_session_cap = st.slider(
        "Physician Session Cap", 10, 25, 18,
        help="Maximum patients per physician per session"
    )
    
    st.sidebar.divider()
    
    # Arrival Parameters
    st.sidebar.subheader("Arrival Rates (patients/hour)")
    
    config.dawn_surge_rate = st.sidebar.slider(
        "Dawn Surge (5-7 AM)", 20, 60, 40,
        help="Arrival rate during early morning surge"
    )
    
    is_monday_default = st.session_state.get('is_monday', False)
    if is_monday_default:
        config.peak_rate_mon = st.sidebar.slider(
            "Peak Rate (Monday)", 80, 140, 110,
            help="Monday peak arrival rate (7-10 AM)"
        )
    else:
        config.peak_rate_normal = st.sidebar.slider(
            "Peak Rate (Non-Monday)", 50, 100, 70,
            help="Non-Monday peak arrival rate (7-10 AM)"
        )
    
    config.midday_rate = st.sidebar.slider(
        "Midday (10 AM-1 PM)", 20, 60, 38,
        help="Midday arrival rate"
    )
    config.afternoon_rate = st.sidebar.slider(
        "Afternoon (1-5 PM)", 10, 40, 20,
        help="Afternoon arrival rate"
    )
    
    st.sidebar.divider()
    
    # Doctor Freeze Parameters
    st.sidebar.subheader("Doctor Staffing Interruptions")
    config.doctor_interruption_frequency = st.sidebar.slider(
        "Interruption Frequency (per day)", 0.5, 3.0, 1.5, 0.5,
        help="How often doctors are pulled to ER/ward duties"
    )
    
    interruption_col1, interruption_col2 = st.sidebar.columns(2)
    with interruption_col1:
        config.doctor_interruption_min_duration = st.slider(
            "Min Duration (min)", 120, 240, 180,
            help="Minimum interruption duration"
        )
    with interruption_col2:
        config.doctor_interruption_max_duration = st.slider(
            "Max Duration (min)", 240, 360, 300,
            help="Maximum interruption duration"
        )
    config.doctor_interruption_mode_duration = st.slider(
        "Mode Duration (min)", 180, 300, 240,
        help="Most likely interruption duration"
    )
    
    st.sidebar.divider()
    
    # Patient Mix
    st.sidebar.subheader("Patient Mix")
    config.priority_patient_percentage = st.sidebar.slider(
        "Priority Patient %", 0.10, 0.40, 0.25,
        format="%.2f",
        help="Percentage of priority patients (Seniors, PWDs, etc.)"
    )
    config.new_patient_percentage = st.sidebar.slider(
        "New Patient %", 0.10, 0.50, 0.30,
        format="%.2f",
        help="Percentage of new (vs returning) patients"
    )
    config.no_show_rate = st.sidebar.slider(
        "No-Show Rate %", 0.00, 0.30, 0.15,
        format="%.2f",
        help="Estimated appointment no-show rate"
    )
    
    st.sidebar.divider()
    
    # Scenario Configuration
    st.sidebar.subheader("Intervention Scenarios")
    st.sidebar.caption("Enable scenarios to test operational improvements")
    
    config.scenario_a_enabled = st.sidebar.checkbox(
        "Scenario A: Front-End Scaling",
        help="Increase verification staff to 3-4 and registration staff to 5-6"
    )
    config.scenario_b_enabled = st.sidebar.checkbox(
        "Scenario B: Consultation Caps",
        help="Enforce physician patient caps and reduce doctor interruptions"
    )
    config.scenario_c_enabled = st.sidebar.checkbox(
        "Scenario C: No-Show Optimization",
        help="Backfill unused appointment slots with walk-in patients"
    )
    config.scenario_d_enabled = st.sidebar.checkbox(
        "Scenario D: Mid-Week Staffing",
        help="Model staffing reductions during Wednesday/Thursday trainings"
    )
    
    return config


def create_kpi_cards(aggregated_results: Dict[str, Any]):
    """Create KPI summary cards."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Avg Total LoS",
            value=f"{aggregated_results.get('los_mean', 0):.1f} min",
            delta=f"±{aggregated_results.get('los_std', 0):.1f}" if aggregated_results.get('los_std') else None
        )
    
    with col2:
        st.metric(
            label="95th Percentile LoS",
            value=f"{aggregated_results.get('los_p95', 0):.0f} min"
        )
    
    with col3:
        completed = int(aggregated_results.get('patients_completed_total', 0))
        arrived = int(aggregated_results.get('patients_arrived_total', 0))
        st.metric(
            label="Throughput",
            value=f"{completed}/{arrived}",
            delta=f"{aggregated_results.get('throughput_rate', 0)*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Doctor Staffing Interruptions",
            value=f"{aggregated_results.get('interruption_events_count', 0):.1f}",
            delta=f"{aggregated_results.get('interruption_total_duration', 0)/60:.1f} hrs total"
        )


def plot_queue_evolution(kpi: KPICollector):
    """Plot queue length evolution over simulation time."""
    
    if not kpi.gate_queue_history:
        st.warning("No queue data available")
        return
    
    # Convert to DataFrame
    times = [t for t, q in kpi.gate_queue_history]
    gate_queue = [q for t, q in kpi.gate_queue_history]
    reg_queue = [q for t, q in kpi.registration_queue_history]
    clinic_queue = [q for t, q in kpi.clinic_queue_history]
    
    # Convert time to hours from 5 AM
    hours = [5 + t/60 for t in times]
    
    df = pd.DataFrame({
        'Hour': hours,
        'Gate Queue': gate_queue,
        'Registration Queue': reg_queue,
        'Clinic Queue': clinic_queue
    })
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Gate Queue Over Time', 'Registration Queue Over Time', 
                        'Clinic Queue Over Time', 'All Queues Comparison'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Gate queue
    fig.add_trace(
        go.Scatter(x=df['Hour'], y=df['Gate Queue'], mode='lines', name='Gate', 
                   line=dict(color='#e74c3c', width=2)),
        row=1, col=1
    )
    
    # Registration queue
    fig.add_trace(
        go.Scatter(x=df['Hour'], y=df['Registration Queue'], mode='lines', name='Registration',
                   line=dict(color='#3498db', width=2)),
        row=1, col=2
    )
    
    # Clinic queue
    fig.add_trace(
        go.Scatter(x=df['Hour'], y=df['Clinic Queue'], mode='lines', name='Clinic',
                   line=dict(color='#2ecc71', width=2)),
        row=2, col=1
    )
    
    # Combined view
    for col, color in [('Gate Queue', '#e74c3c'), ('Registration Queue', '#3498db'), ('Clinic Queue', '#2ecc71')]:
        fig.add_trace(
            go.Scatter(x=df['Hour'], y=df[col], mode='lines', name=col, line=dict(color=color, width=2)),
            row=2, col=2
        )
    
    # Add vertical lines for key times
    for hour, label in [(8, 'OPD Opens'), (12, 'Midday'), (17, 'OPD Closes')]:
        fig.add_vline(x=hour, line_dash="dash", line_color="gray", row="all", col="all")
        fig.add_annotation(x=hour, y=max(clinic_queue)*0.9, text=label, showarrow=False,
                          textangle=-90, font_size=10, row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Queue Length Evolution Throughout the Day",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_yaxes(title_text="Queue Length", row=1, col=1)
    fig.update_yaxes(title_text="Queue Length", row=2, col=1)
    fig.update_yaxes(title_text="Queue Length", row=2, col=2)
    
    st.plotly_chart(fig, width="stretch")


def plot_waiting_times(aggregated_results: Dict[str, Any]):
    """Plot waiting times by stage."""
    
    stages = ['Gate & Verification', 'Registration', 'Clinic', 'Consultation']
    means = [
        aggregated_results.get('gate_verify_vitals_mean', 0),
        aggregated_results.get('registration_wait_mean', 0),
        aggregated_results.get('clinic_wait_mean', 0),
        aggregated_results.get('consultation_wait_mean', 0)
    ]
    stds = [
        aggregated_results.get('gate_verify_vitals_std', 0),
        aggregated_results.get('registration_wait_std', 0),
        aggregated_results.get('clinic_wait_std', 0),
        0
    ]
    p95s = [
        aggregated_results.get('gate_verify_vitals_p95', 0),
        aggregated_results.get('registration_wait_p95', 0),
        aggregated_results.get('clinic_wait_p95', 0),
        aggregated_results.get('consultation_wait_mean', 0) * 1.5
    ]
    
    df = pd.DataFrame({
        'Stage': stages,
        'Mean Wait (min)': means,
        'Std Dev': stds,
        '95th Percentile': p95s
    })
    
    fig = px.bar(
        df, x='Stage', y=['Mean Wait (min)', '95th Percentile'],
        barmode='group',
        title='Average Waiting Times by Stage',
        color_discrete_map={'Mean Wait (min)': '#3498db', '95th Percentile': '#e74c3c'}
    )
    
    fig.update_layout(
        yaxis_title="Waiting Time (minutes)",
        legend_title="Metric",
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")


def plot_length_of_stay_distribution(kpi: KPICollector):
    """Plot length of stay histogram."""
    
    if not kpi.length_of_stay:
        st.warning("No length of stay data available")
        return
    
    los_hours = [los / 60 for los in kpi.length_of_stay]  # Convert to hours
    
    fig = px.histogram(
        x=los_hours,
        nbins=30,
        title='Distribution of Total OPD Length of Stay',
        labels={'x': 'Length of Stay (hours)', 'y': 'Number of Patients'},
        color_discrete_sequence=['#3498db']
    )
    
    # Add mean line
    mean_los = np.mean(los_hours)
    fig.add_vline(x=mean_los, line_dash="dash", line_color="#e74c3c", 
                  annotation_text=f"Mean: {mean_los:.2f} hrs")
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def plot_consultation_service_time(kpi: KPICollector):
    """Plot consultation service time distribution."""
    
    if not kpi.consultation_service_times:
        st.warning("No consultation service time data available")
        return
    
    fig = px.histogram(
        x=kpi.consultation_service_times,
        nbins=25,
        title='Consultation Service Time Distribution',
        labels={'x': 'Service Time (minutes)', 'y': 'Number of Consultations'},
        color_discrete_sequence=['#2ecc71']
    )
    
    # Add statistics
    mean_time = np.mean(kpi.consultation_service_times)
    fig.add_vline(x=mean_time, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"Mean: {mean_time:.1f} min")
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def plot_doctor_interruption_analysis(kpi: KPICollector):
    """Visualize doctor freeze events."""
    
    if not kpi.doctor_interruption_events:
        st.info("No doctor staffing interruptions recorded in this simulation run")
        return
    
    df = pd.DataFrame(kpi.doctor_interruption_events)
    df['end_time'] = df['start_time'] + df['duration']
    df['start_hour'] = 5 + df['start_time'] / 60
    df['end_hour'] = 5 + df['end_time'] / 60
    
    fig = px.timeline(
        df, x_start='start_hour', x_end='end_hour', y='clinic',
        color='clinic',
        title='Doctor Staffing Interruptions Throughout the Day',
        labels={'start_hour': 'Hour', 'clinic': 'Clinic'}
    )
    
    fig.update_layout(
        xaxis_title="Hour of Day (starting from 5 AM)",
        yaxis_title="Clinic",
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Interruption Events", len(kpi.doctor_interruption_events))
    with col2:
        total_duration = sum(e['duration'] for e in kpi.doctor_interruption_events)
        st.metric("Total Interruption Duration", f"{total_duration/60:.1f} hours")
    with col3:
        avg_duration = total_duration / len(kpi.doctor_interruption_events) if kpi.doctor_interruption_events else 0
        st.metric("Average Interruption Duration", f"{avg_duration:.0f} minutes")


def plot_resource_utilization(kpi: KPICollector):
    """Display resource utilization metrics."""
    
    resources = ['Gate Security', 'Verification Staff', 'Health Aides', 
                 'Registration Staff', 'Clinic Routing', 'Sub-Clinic Nursing', 'OPD Physicians']
    
    # Map display names to KPI keys
    resource_keys = {
        'Gate Security': 'gate_security',
        'Verification Staff': 'verification_staff',
        'Health Aides': 'health_aides',
        'Registration Staff': 'registration_staff',
        'Clinic Routing': 'clinic_routing',
        'Sub-Clinic Nursing': 'sub_clinic_nursing',
        'OPD Physicians': 'physicians'
    }
    
    utilization_list = []
    for r in resources:
        key = resource_keys[r]
        busy = kpi.resource_busy_time.get(key, 0.0)
        total = kpi.resource_total_time.get(key, 0.0)
        if total > 0:
            util = (busy / total) * 100
        else:
            util = 0.0
        utilization_list.append(min(100.0, util))
    
    utilization_df = pd.DataFrame({
        'Resource': resources,
        'Utilization (%)': utilization_list
    })
    
    fig = px.bar(
        utilization_df, x='Resource', y='Utilization (%)',
        title='Resource Utilization Rates',
        color='Utilization (%)',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        range_color=[0, 100]
    )
    
    # Add threshold line
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="80% Target", annotation_position="top right")
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def plot_patient_throughput(kpi: KPICollector):
    """Display patient throughput by routing outcome."""
    
    # Analyze patient records
    if not kpi.patient_records:
        st.warning("No patient records available")
        return
    
    df = pd.DataFrame(kpi.patient_records)
    
    # By patient type
    type_counts = df['patient_type'].value_counts()
    
    # By clinic
    clinic_counts = df['clinic'].value_counts()
    
    # By exit reason
    exit_counts = df['exit_reason'].value_counts() if 'exit_reason' in df.columns else {}
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('By Patient Source', 'By Clinic', 'By Exit Outcome'),
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]]
    )
    
    fig.add_trace(
        go.Pie(labels=type_counts.index, values=type_counts.values, 
               name="Patient Source", hole=0.4),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(labels=clinic_counts.index, values=clinic_counts.values,
               name="Clinic", hole=0.4),
        row=1, col=2
    )
    
    if len(exit_counts) > 0:
        fig.add_trace(
            go.Pie(labels=exit_counts.index, values=exit_counts.values,
                   name="Exit Outcome", hole=0.4),
            row=1, col=3
        )
    
    fig.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig, width="stretch")


def plot_pre_opening_queue(kpi: KPICollector):
    """Plot pre-opening queue build-up (5 AM to 8 AM)."""
    
    if not kpi.pre_opening_queue:
        # Generate sample data if not available
        st.info("Pre-opening queue data being recorded...")
        return
    
    times = [t for t, q in kpi.pre_opening_queue]
    counts = [q for t, q in kpi.pre_opening_queue]
    
    # Convert to hours from 5 AM
    hours = [5 + t/60 for t in times]
    
    fig = px.line(
        x=hours, y=counts,
        title='Pre-Opening Queue Build-up (5 AM - 8 AM)',
        labels={'x': 'Hour of Day', 'y': 'Patients in Queue'},
        color_discrete_sequence=['#9b59b6']
    )
    
    fig.update_layout(height=350)
    fig.add_vline(x=8, line_dash="dash", line_color="green", 
                  annotation_text="OPD Opens", annotation_position="top right")
    
    st.plotly_chart(fig, width="stretch")


def display_scenario_comparison(stored_results: Dict[str, Dict]):
    """Display comparison of saved scenario runs."""
    
    if len(stored_results) < 2:
        st.info("Run multiple scenarios and save them to enable comparison")
        return
    
    scenarios = list(stored_results.keys())
    
    comparison_df = pd.DataFrame({
        'Scenario': scenarios,
        'Avg LoS (min)': [stored_results[s].get('los_mean', 0) for s in scenarios],
        '95th LoS (min)': [stored_results[s].get('los_p95', 0) for s in scenarios],
        'Throughput': [f"{stored_results[s].get('throughput_rate', 0)*100:.1f}%" for s in scenarios],
        'Gate & Verification Wait (min)': [stored_results[s].get('gate_verify_vitals_mean', 0) for s in scenarios],
        'Reg Wait (min)': [stored_results[s].get('registration_wait_mean', 0) for s in scenarios],
        'Clinic Wait (min)': [stored_results[s].get('clinic_wait_mean', 0) for s in scenarios],
        'Staffing Interruptions': [stored_results[s].get('interruption_events_count', 0) for s in scenarios]
    })
    
    st.subheader("Scenario Comparison")
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    
    csv = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download comparison CSV", csv, "scenario_comparison.csv", "text/csv")
    
    # Visual comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Length of Stay', 'Waiting Times', 
                        'Throughput Rate', 'Doctor Staffing Interruptions')
    )
    
    # LoS comparison
    fig.add_trace(
        go.Bar(x=scenarios, y=[stored_results[s].get('los_mean', 0) for s in scenarios],
               marker_color='#3498db', name='Avg LoS'),
        row=1, col=1
    )
    
    # Waiting times comparison
    wait_configs = [
        ('Gate & Verification Wait', 'gate_verify_vitals_mean', '#e74c3c'),
        ('Reg Wait', 'registration_wait_mean', '#2ecc71'),
        ('Clinic Wait', 'clinic_wait_mean', '#9b59b6')
    ]
    for label, key, color in wait_configs:
        fig.add_trace(
            go.Bar(x=scenarios, 
                   y=[stored_results[s].get(key, 0) for s in scenarios],
                   name=label, marker_color=color),
            row=1, col=2
        )
    
    # Throughput comparison
    fig.add_trace(
        go.Bar(x=scenarios, 
               y=[stored_results[s].get('throughput_rate', 0)*100 for s in scenarios],
               marker_color='#f39c12', name='Throughput %'),
        row=2, col=1
    )
    
    # Freeze events comparison
    fig.add_trace(
        go.Bar(x=scenarios, 
               y=[stored_results[s].get('interruption_events_count', 0) for s in scenarios],
               marker_color='#e67e22', name='Staffing Interruptions'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, 
                      title_text="Multi-Scenario Performance Comparison")
    
    st.plotly_chart(fig, width="stretch")


# ============================================================================
# SECTION 6: MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="DRMC OPD Simulation Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .stMetric label {
        font-weight: 600;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("DRMC Outpatient Department Patient Flow Simulation")
    st.markdown("""
    **Discrete Event Simulation (DES) Model for Davao Regional Medical Center OPD Operations**
    
    This interactive dashboard simulates patient flow through the DRMC OPD, modeling:
    - Pre-opening queue formation (starting 5:00 AM)
    - Gate screening, verification & vitals capture
    - Central registration (new vs returning patients)
    - Sub-clinic routing and consultation
    - Post-consultation ancillary services
    - Doctor staffing interruptions (ER/ward duty calls)
    """)
    
    st.divider()
    
    # Initialize session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'stored_scenarios' not in st.session_state:
        st.session_state.stored_scenarios = {}
    if 'config' not in st.session_state:
        st.session_state.config = SimulationConfig()
    
    # Create sidebar configuration
    config = create_sidebar_config()
    st.session_state.config = config
    
    # Baseline Scenario Shortcut
    st.sidebar.divider()
    st.sidebar.subheader("Baseline Run Control")
    if st.sidebar.button("Save as Baseline", use_container_width=True, help="Lock current configuration and run/save as Baseline scenario"):
        if st.session_state.simulation_results is not None:
            st.session_state.stored_scenarios["Baseline"] = (
                st.session_state.simulation_results['aggregated']
            )
            st.sidebar.success("Baseline saved from current run!")
        else:
            with st.sidebar.status("Running baseline simulation...") as status:
                kpi_results = run_simulation(config)
                aggregated = aggregate_results([kpi_results])
                st.session_state.stored_scenarios["Baseline"] = aggregated
                st.session_state.simulation_results = {
                    'kpi': kpi_results,
                    'aggregated': aggregated,
                    'config': config
                }
                status.update(label="Baseline simulated and saved!", state="complete")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "Run Simulation", 
        "Results & Analytics", 
        "Scenario Comparison",
        "Configuration Summary"
    ])
    
    with tab1:
        st.header("Simulation Control Panel")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Run Configuration")
            
            # Display current config summary
            config_summary = f"""
            **Daily Volume:** {config.daily_volume} patients
            
            **Staff Configuration:**
            - Gate Security: {config.gate_security_count}
            - Verification Staff: {config.verification_staff_count}
            - Registration Staff: {config.registration_staff_count}
            - Physicians (Morning): {config.morning_physician_count}
            - Physicians (Afternoon): {config.afternoon_physician_count}
            
            **Arrival Rates:**
            - Dawn Surge: {config.dawn_surge_rate}/hr
            - Peak (Mon): {config.peak_rate_mon if config.simulation_day == 1 else config.peak_rate_normal}/hr
            - Midday: {config.midday_rate}/hr
            - Afternoon: {config.afternoon_rate}/hr
            
            **Active Scenarios:**
            {'- Scenario A: Front-End Scaling' if config.scenario_a_enabled else ''}
            {'- Scenario B: Consultation Caps' if config.scenario_b_enabled else ''}
            {'- Scenario C: No-Show Optimization' if config.scenario_c_enabled else ''}
            {'- Scenario D: Mid-Week Staffing' if config.scenario_d_enabled else ''}
            """
            st.markdown(config_summary)
            
        with col2:
            st.subheader("Quick Actions")
            
            run_button = st.button("Run Simulation", type="primary", width="stretch")
            
            scenario_name = st.text_input("Scenario Name (for saving)", 
                                          placeholder="e.g., Baseline, Scenario A, etc.")
            
            save_button = st.button("Save Current Results", width="stretch")
            
            clear_button = st.button("Clear All Saved Scenarios", width="stretch")
        
        st.divider()
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run simulation
        if run_button:
            status_text.text("Initializing simulation...")
            progress_bar.progress(10)
            
            try:
                # Run simulation with progress callback
                def update_progress(progress):
                    progress_bar.progress(10 + int(progress * 80))
                    status_text.text(f"Running replication {int(progress * config.num_replications) + 1} of {config.num_replications}...")
                
                status_text.text("Running simulation...")
                kpi_results = run_simulation(config, progress_callback=update_progress)
                
                progress_bar.progress(90)
                status_text.text("Aggregating results...")
                
                # Aggregate results
                aggregated = aggregate_results([kpi_results])
                
                progress_bar.progress(100)
                status_text.text("Simulation complete!")
                
                # Store results
                st.session_state.simulation_results = {
                    'kpi': kpi_results,
                    'aggregated': aggregated,
                    'config': config
                }
                
                st.success(f"Simulation completed! Processed {int(aggregated['patients_completed_total'])} patients.")
                
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                progress_bar.progress(0)
        
        # Save scenario
        if save_button and st.session_state.simulation_results:
            if scenario_name:
                st.session_state.stored_scenarios[scenario_name] = st.session_state.simulation_results['aggregated']
                st.success(f"Scenario '{scenario_name}' saved!")
            else:
                st.warning("Please enter a scenario name before saving.")
        
        # Clear scenarios
        if clear_button:
            st.session_state.stored_scenarios = {}
            st.info("All saved scenarios cleared.")
    
    with tab2:
        st.header("Results & Analytics Dashboard")
        
        if st.session_state.simulation_results:
            results = st.session_state.simulation_results
            kpi = results['kpi']
            aggregated = results['aggregated']
            
            # KPI Summary Cards
            st.subheader("Key Performance Indicators")
            create_kpi_cards(aggregated)
            
            st.divider()
            
            # Queue Analysis
            st.subheader("Queue Dynamics Analysis")
            
            queue_col1, queue_col2 = st.columns(2)
            with queue_col1:
                plot_pre_opening_queue(kpi)
            with queue_col2:
                st.metric("Max Gate Queue", f"{aggregated.get('gate_queue_max', 0):.0f} patients")
                st.metric("Max Registration Queue", f"{aggregated.get('registration_queue_max', 0):.0f} patients")
                st.metric("Max Clinic Queue", f"{aggregated.get('clinic_queue_max', 0):.0f} patients")
            
            plot_queue_evolution(kpi)
            
            st.divider()
            
            # Waiting Time Analysis
            st.subheader("Waiting Time Analysis by Stage")
            plot_waiting_times(aggregated)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Gate & Verification Wait", f"{aggregated.get('gate_verify_vitals_mean', 0):.1f} min")
            with col2:
                st.metric("Avg Registration Wait", f"{aggregated.get('registration_wait_mean', 0):.1f} min")
            with col3:
                st.metric("Avg Clinic Wait", f"{aggregated.get('clinic_wait_mean', 0):.1f} min")
            
            st.divider()
            
            # Service Time and LoS
            st.subheader("Service Time & Length of Stay")
            
            los_col, consult_col = st.columns(2)
            with los_col:
                plot_length_of_stay_distribution(kpi)
            with consult_col:
                plot_consultation_service_time(kpi)
            
            st.divider()
            
            # Doctor Freeze Analysis
            st.subheader("Doctor Staffing Interruption Analysis")
            plot_doctor_interruption_analysis(kpi)
            
            st.divider()
            
            # Resource Utilization
            st.subheader("Resource Utilization")
            plot_resource_utilization(kpi)
            
            st.divider()
            
            # Patient Throughput
            st.subheader("Patient Throughput Analysis")
            plot_patient_throughput(kpi)
            
        else:
            st.info("Run a simulation first to see results!")
            
            # Show sample/placeholder charts
            st.subheader("Sample Visualization Preview")
            st.markdown("""
            **Charts that will be generated after running simulation:**
            
            1. **Queue Evolution** - Time-series showing queue lengths at Gate, Registration, and Clinic throughout the day
            
            2. **Waiting Times by Stage** - Bar chart comparing average wait times across all stages
            
            3. **Length of Stay Distribution** - Histogram of total patient time in system
            
            4. **Consultation Service Times** - Distribution of time spent with physicians
            
            5. **Doctor Staffing Interruption Timeline** - Gantt chart showing when doctors are pulled for emergency/ward duties
            
            6. **Resource Utilization** - Bar chart showing busy time percentage per staff type
            
            7. **Patient Throughput** - Pie charts showing patient distribution by source, clinic, and outcome
            """)
    
    with tab3:
        st.header("Scenario Comparison")
        
        display_scenario_comparison(st.session_state.stored_scenarios)
        
        if st.session_state.stored_scenarios:
            st.divider()
            st.subheader("Saved Scenarios Summary")
            
            for name, data in st.session_state.stored_scenarios.items():
                with st.expander(f"{name}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg LoS", f"{data.get('los_mean', 0):.1f} min")
                    with col2:
                        st.metric("Throughput", f"{data.get('throughput_rate', 0)*100:.1f}%")
                    with col3:
                        st.metric("Total Patients", f"{int(data.get('patients_completed_total', 0))}")
    
    with tab4:
        st.header("Configuration Summary")
        
        st.subheader("Current Simulation Parameters")
        
        config = st.session_state.config
        
        # Create detailed config display
        config_data = {
            'Category': [],
            'Parameter': [],
            'Value': []
        }
        
        # Volume
        config_data['Category'].extend(['Volume'] * 2)
        config_data['Parameter'].extend(['Daily Patient Volume', 'Replications'])
        config_data['Value'].extend([config.daily_volume, config.num_replications])
        
        # Staffing
        config_data['Category'].extend(['Staffing'] * 9)
        config_data['Parameter'].extend([
            'Gate Security', 'Verification Staff', 'Health Aides', 
            'Registration Staff', 'Clinic Routing', 'Sub-Clinic Nursing',
            'Morning Physicians', 'Afternoon Physicians', 'Physician Cap'
        ])
        config_data['Value'].extend([
            config.gate_security_count, config.verification_staff_count, config.health_aide_count,
            config.registration_staff_count, config.clinic_routing_staff_count, config.sub_clinic_nursing_count,
            config.morning_physician_count, config.afternoon_physician_count, config.physician_session_cap
        ])
        
        # Arrival rates
        config_data['Category'].extend(['Arrival Rates'] * 4)
        config_data['Parameter'].extend(['Dawn Surge', 'Peak Rate', 'Midday', 'Afternoon'])
        config_data['Value'].extend([config.dawn_surge_rate, config.peak_rate_normal, config.midday_rate, config.afternoon_rate])
        
        # Interruption parameters
        config_data['Category'].extend(['Staffing Interruptions'] * 4)
        config_data['Parameter'].extend(['Frequency', 'Min Duration', 'Mode Duration', 'Max Duration'])
        config_data['Value'].extend([
            config.doctor_interruption_frequency, config.doctor_interruption_min_duration,
            config.doctor_interruption_mode_duration, config.doctor_interruption_max_duration
        ])
        
        # Patient mix
        config_data['Category'].extend(['Patient Mix'] * 3)
        config_data['Parameter'].extend(['Priority %', 'New Patient %', 'No-Show Rate'])
        config_data['Value'].extend([
            f"{int(config.priority_patient_percentage * 100)}%",
            f"{int(config.new_patient_percentage * 100)}%",
            f"{int(config.no_show_rate * 100)}%"
        ])
        
        df = pd.DataFrame(config_data)
        df['Value'] = df['Value'].astype(str)
        st.dataframe(df, width="stretch", hide_index=True)
        
        st.divider()
        
        # Scenario status
        st.subheader("Active Intervention Scenarios")
        
        scenario_status = {
            'Scenario': ['A: Front-End Scaling', 'B: Consultation Caps', 
                         'C: No-Show Optimization', 'D: Mid-Week Staffing'],
            'Enabled': [
                'Yes' if config.scenario_a_enabled else 'No',
                'Yes' if config.scenario_b_enabled else 'No',
                'Yes' if config.scenario_c_enabled else 'No',
                'Yes' if config.scenario_d_enabled else 'No'
            ],
            'Effect': [
                'Increases verification to 3-4, registration to 5-6',
                'Enforces 15-20 patient cap per session',
                f'Backfills {int(config.no_show_rate * 100)}% no-show slots',
                'Reduces staff 25% on Wed/Thu'
            ]
        }
        
        st.dataframe(pd.DataFrame(scenario_status), width="stretch", hide_index=True)
        
        st.divider()
        
        # Distribution parameters reference
        st.subheader("Probability Distributions Used")
        
        dist_data = {
            'Process': ['Gate Verification', 'New Registration', 'Returning Registration', 
                        'Consultation Service', 'Doctor Interruption Duration'],
            'Distribution': ['Triangular', 'Triangular', 'Triangular', 'Triangular', 'Triangular'],
            'Parameters (a, b, c)': [
                '(0.5, 2.5, 15) min',
                '(7, 17.5, 60) min',
                '(1, 4, 20) min',
                '(5, 15, 45) min',
                f'({config.doctor_interruption_min_duration}, {config.doctor_interruption_mode_duration}, {config.doctor_interruption_max_duration}) min'
            ],
            'Source': ['Proposal Doc', 'Proposal Doc', 'Proposal Doc', 'Proposal Doc', 'Proposal Doc']
        }
        
        st.dataframe(pd.DataFrame(dist_data), width="stretch", hide_index=True)
        
        st.divider()
        
        # Footer information
        st.markdown("""
        ---
        **DRMC OPD Simulation Model**
        
        Built with: SimPy (Discrete Event Simulation), Streamlit (Web UI), Plotly (Visualization)
        
        Based on: DRMC_OPD_Simulation_Proposal.pdf
        
        For questions or details, refer to the project documentation.
        """)
    
    # Modeling Assumptions Expander
    st.markdown("---")
    with st.expander("Modeling Assumptions"):
        st.markdown("""
        The simulation model is constructed based on the following key operational assumptions:
        - **Pooled Poisson Arrivals**: Arrivals are pooled Poisson (no per-clinic differentiation).
        - **Consultation Service Time**: Consultation service time is Triangular(5, 15, 45) minutes.
        - **Doctor Staffing Interruptions**: Emergency or ward duties temporarily reduce physician capacity to 75% for Triangular(180, 240, 300) minutes.
        - **Priority Queue Preemption**: Priority patients preempt the queue but do not interrupt active consultations.
        - **Ancillary Services**: We do not model diagnostic/pharmacy turnaround in detail (out of scope).
        - **Rescheduling**: We do not model late-arrival rescheduling (out of scope).
        """)


if __name__ == "__main__":
    main()