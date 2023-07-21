import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import plotly as py
import plotly.tools
import matplotlib
matplotlib.use('agg')
sys.setrecursionlimit(10000)


class G:
    ITERATIONS = 100
    DURATIONS = 6120

    
G.target_times = {
    1: 2,
    2: 3,
    3: 5
}


def update_target_times(target_times):
    G.target_times = {
        1: target_times[0],
        2: target_times[1],
        3: target_times[2]
    }

    
G.specialties = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5'
}


def update_globals(urg_times):
    G.process_times = {
                        1: urg_times[0],
                        2: urg_times[1],
                        3: urg_times[2]
                    }    
    

def create_arrival_times(sim_time, arr_rates):  #[time_between_urg 1 images, etc..]
    arrival_times_dict = {}
    arrival_times_tuples_list = []
    #Create arrival times for each urgency of images
    urg = 1
    for arr_time in arr_rates:
        arrival_times = []
        time = 0
        while time < sim_time:
            time += np.random.exponential(arr_time)
            arrival_times.append(time)
            arrival_times_tuples_list.append([urg, time])
        arrival_times_dict[urg] = arrival_times
        urg += 1
    arrival_times_tuples_list = sorted(arrival_times_tuples_list, key=lambda x: x[1])
    return arrival_times_dict, arrival_times_tuples_list


def create_medical_images(arrival_times_tuples_list):
    med_images = []
    for img_id, tup in enumerate(arrival_times_tuples_list):
        med_images.append(MedicalImage(img_id, tup[1], tup[0], random.sample(list(G.specialties.keys()), 1)[0]))
    print(f"{len(med_images)} medical images")
    return med_images


def create_constant_rads(num_rads):
    specialties_list = []
    for i in range(num_rads):
        specialties_list.append(random.sample(list(G.specialties.keys()), random.randrange(2,len(G.specialties))))
    G.const_specialities = specialties_list

    
def create_radiologists(num_rads, constant_rads):
    radiologists = []
    for i in range(num_rads):
        if constant_rads:
            specialties_temp = G.const_specialities[i] 
        else:
            specialties_temp = random.sample(list(G.specialties.keys()), random.randrange(2,len(G.specialties)))
        radiologists.append(Radiologist(i, specialties_temp))
    return radiologists


def create_initial_events(sim_duration, med_images, cutoff=False):
    events=[]
    for img in med_images:
        events.append([img.time_created, 'New Job', img])
    if cutoff:
        events.append([sim_duration*2, "Sim End"])
    events = sorted(events, key=lambda x: x[0])
    return events


def start_simulation(events, med_images, radiologists, constant_rads, cutoff=False):
    s = SystemState(events, med_images, G.radiologists, cutoff)
    s.run_simulation()


class MedicalImage(object):    
    def __init__(self, img_id, time_created, urgency, image_type):#, modality, speciality, urgency, image_label):
        self.img_id = img_id
        self.time_created = time_created
        self.urgency = urgency
        self.image_type = image_type
        self.target_time = G.target_times[urgency]
        self.time_remaining = G.target_times[urgency]
        self.est_process_time = G.process_times[urgency]
        self.in_queues = []   #keep track on which queues image is in [rad_id, position]
        self.time_seen = 0
        self.time_done = 0
        self.rad_seen = "None"
        
    def update_time_remaining(self, t):
        self.time_remaining = self.target_time - (t - self.time_created)
        
        
class Radiologist:
    def __init__(self, rad_id, specialties, working=True):
        self.queue = []
        self.queue_data = []#[med_image, image_id, image_urgency, time_left, est_time]
        self.rad_id = rad_id
        self.specialties = specialties
        self.is_working = working
        self.is_idle = 1
        self.images_served = []
        self.idle_times = []
        self.time_busy_start = 0
        self.time_idle_start = 0
        self.busy_times = []
        self.time = 0
        self.time_of_step = 0
        self.queue_length = []
        self.service_starts = []
        self.service_ends = []
        self.service_time = []  
        
    def get_stats(self):
        return self.idle_times, self.busy_times, self.queue_length, self.service_starts, self.service_ends, self.service_time 
        
    def show_queue(self):
        return self.queue
    
    def estimate_queue_time(self):
        t = 0
        for img in self.queue:
            t += img.est_process_time
        return t
    
    def add_job(self, med_image, time):
        #update idle time tracker
        if self.is_idle == 1:
            self.idle_times.append(time - self.time_idle_start)
            self.time_busy_start = time
        self.is_idle = 0
        #add img to queue and sort
        #self.queue.append(med_image)
        #if len(self.queue) > 1:
        #    self.sort_queue()
        if len(self.queue) <= 1:
            self.queue.append(med_image)
            self.queue_data.append([med_image, med_image.img_id, med_image.urgency, med_image.time_remaining, med_image.est_process_time, med_image.est_process_time]) #[image_id, image_urgency, time_left, est_time]
        else:
            if med_image.urgency == 1:
                ins_ind = 1
                for ind, img in enumerate(self.queue[1:]):
                    if img.urgency > 1:
                        ins_ind = ind + 1
                        break          
                self.queue.insert(ins_ind, med_image)
            elif med_image.urgency == 2:
                ins_ind = 1
                for ind, img in enumerate(self.queue[1:]):
                    if img.urgency > 2:
                        ins_ind = ind + 1
                        break          
                self.queue.insert(ins_ind, med_image)
            else:
                self.queue.append(med_image)

    def sort_queue(self):
        curr_img = self.queue[0]
        queue_tuple = [[img, img.urgency] for img in self.queue[1:]]
        queue_tuple = sorted(queue_tuple, key=lambda x: x[1])
        #print(queue_tuple)
        queue_tuple_list = [tup[0] for tup in queue_tuple]
        self.queue = queue_tuple_list
        self.queue.insert(0, curr_img)
       
    def finish_job(self, time):
        if len(self.queue) == 0:
            self.time_finished_last_job = time
            self.time_idle_start = time
            self.busy_times.append(time - self.time_busy_start)
            self.is_idle = 1
    
    def update_idle_lists(self, time):
        if self.is_idle == 1:
            self.idle_times.append(time - self.time_idle_start)
        elif self.is_idle == 0:
            self.busy_times.append(time - self.time_busy_start)
        
    def update_queue(self, time):
        for img in self.queue:
            img.update_time_remaining(time)
        #sort_queue()        
    #def sort_queue(self):
                   
        
        
class SystemState:
    def __init__(self, sim_duration, events, images, rads, cutoff=False, verbose=False):
        self.time = 0
        self.sim_duration = sim_duration
        self.continue_running = True
        self.events = events
        self.images = images
        self.rads = rads
        self.rads_working = rads
        self.rads_not_working = []
        self.events_history = []
        self.queue_lengths = []
        self.time_steps = []
        self.img_table = pd.DataFrame(columns=['img_id','urgency', 'rad_id', 'time_created','time_rad_job_starts', 'time_job_finished', 'wait_time', 'time_w_rad', 'total_time'])
        self.rad_table = pd.DataFrame()
        self.unfin_img_table = pd.DataFrame(columns=['img_id','urgency', 'rad_id', 'time_created','time_rad_job_starts', 'time_job_finished', 'wait_time', 'time_w_rad', 'total_time'])
        self.verbose = verbose
        
    def create_event(self, time, event_type, obj):
        self.events.append([time, event_type, obj])
        self.events = sorted(self.events, key=lambda x: x[0])

    def update_img_table(self, med_img):
        column_names = ['img_id','urgency', 'rad_id', 'time_created','time_rad_job_starts', 'time_job_finished', 'wait_time', 'time_w_rad', 'total_time']
        values = [med_img.img_id, med_img.urgency, med_img.rad_seen, med_img.time_created, med_img.time_seen, self.time, med_img.time_seen - med_img.time_created, self.time - med_img.time_seen, self.time - med_img.time_created] #[[med_img.time_created], [med_img.time_seen], [self.time], [self.time - med_img.time_seen], [self.time - med_img.time_created]]
        temp_df = pd.DataFrame(values).T
        temp_df.columns = column_names
        #self.img_table = self.img_table.append(temp_df, ignore_index = True)
        self.img_table = pd.concat([self.img_table, temp_df], ignore_index=True)
        
    def unfinished_jobs(self):
        unfin_med_images = []
        for rad in self.rads:
            unfin_med_images += rad.queue
            unfin_med_images = list(set(unfin_med_images))
        print(f"There are {len(unfin_med_images)} that were not completed in time")
        column_names = [['img_id','urgency', 'rad_id', 'time_created','time_rad_job_starts', 'time_job_finished', 'wait_time', 'time_w_rad', 'total_time']]
        for med_img in unfin_med_images:
            values = [med_img.img_id, med_img.urgency, med_img.rad_seen, med_img.time_created, med_img.time_seen, self.time, med_img.time_seen - med_img.time_created, self.time - med_img.time_seen, self.time - med_img.time_created] #[[med_img.time_created], [med_img.time_seen], [self.time], [self.time - med_img.time_seen], [self.time - med_img.time_created]]
            temp_df = pd.DataFrame(values).T
            temp_df.columns = column_names
            self.unfin_img_table = pd.concat([self.unfin_img_table, temp_df], ignore_index = True)
        
    def process_event(self):
        event = self.events[0]
        self.events_history.append(event)
        self.time = event[0]       
        event_type = event[1]
        del self.events[0]
        temp_list = []
        for r in self.rads:
            temp_list.append(len(r.queue))
        self.queue_lengths.append(temp_list)
        self.time_steps.append(self.time)        
            
        if event_type == "New Job":
            self.distribute_job(event[2])
        elif event_type == "Job Done":
            rad = event[2]
            self.complete_job(rad)
        elif event_type == "Sim End":
            self.continue_running = False 
        if self.verbose==True:
            print("Event processed")
        if (len(self.events) == 0) or (self.events[0][1]=="Sim End"):
            self.continue_running = False 
        if self.continue_running:
            self.process_event()
        else:
            for rad in self.rads:
                rad.update_idle_lists(self.time)
            self.unfinished_jobs()
            print(f"Simulation complete at {self.time} minutes")
                
    def distribute_job(self, med_image):
        urgency = med_image.urgency
        image_type = med_image.image_type
        # Function to route medical images based on some algorithm
        chosen_rads = self.choose_rads(image_type)       
        for rad in chosen_rads:
            rad.add_job(med_image, self.time)
            med_image.in_queues.append(rad)    #keep track of which rads have image in queue
            if len(rad.queue)==1:
                self.start_job(rad)
                break         
        self.update_queues() 
        
    def choose_rads(self, image_type):
        capable_rads = []
        for rad in self.rads_working:      #finds radiologists capable of working on image
            if image_type in rad.specialties:
                capable_rads.append(rad)
        chosen_rads = self.n_quickest_queues(capable_rads, 3)
        return capable_rads
    
    def n_shortest_queues(self, rads_list, n):
        rads_tuples = []
        for rad in rads_list:
            rads_tuples.append([rad, len(rad.queue)])
        rads_tuples.sort(key = lambda x: x[1])
        return [rad[0] for rad in rads_tuples[:n]]
    
    def n_quickest_queues(self, rads_list, n):
        rads_tuples = []
        for rad in rads_list:
            rads_tuples.append([rad, rad.estimate_queue_time()])
        rads_tuples.sort(key = lambda x: x[1])
        return [rad[0] for rad in rads_tuples[:n]]
             
    def update_queues(self):
        for rad in self.rads_working:
            rad.update_queue(self.time)
                
    def start_job(self, rad):
        med_image = rad.queue[0]
        image_type = med_image.image_type
        urgency = med_image.urgency
        rad.service_starts = self.time
        med_image.time_seen = self.time
        med_image.rad_seen = rad.rad_id
        self.events_history.append([self.time, "Job Started", med_image])
        process_time = np.random.exponential(G.target_times[urgency])
        self.create_event(self.time+process_time, "Job Done", rad)
        if self.verbose==True:
            print(f"Image {med_image.img_id} is seen by radiologist {rad.rad_id} at {self.time}")
        for r in med_image.in_queues:
            if r != rad:
                r.queue.remove(med_image)           
        
    def complete_job(self, rad):
        med_image = rad.queue[0]
        self.update_img_table(med_image)
        rad.images_served.append(med_image.img_id)
        rad.service_ends.append(self.time)
        med_image.time_done = self.time
        if self.verbose==True:
            print(f"Image {med_image.img_id} is done by radiologist {rad.rad_id} at {self.time}")
        del rad.queue[0]
        rad.finish_job(self.time)
        if len(rad.queue) > 0:
            self.start_job(rad)

    def run_simulation(self):
        self.process_event()

def gen_system_state(sim_time, rads_count, arr_rates, urg_times, constant_rads, cutoff, verbose):
    #Define urgency times
    update_globals(urg_times)
    #Create the intervals
    arrivals_dict, arrival_times_tuples_list = create_arrival_times(sim_time, arr_rates)
    #Create the images with their arrival time_seen
    med_images = create_medical_images(arrival_times_tuples_list)
    #Create the radiologists
    radiologists = create_radiologists(rads_count, constant_rads)
    #Create the image arrival events
    events = create_initial_events(sim_time, med_images, cutoff)
    s = SystemState(sim_time, events, med_images, radiologists, cutoff, verbose)
    return s


def sim(sim_time, rads_count, arr_rates, urg_times, constant_rads=False, cutoff=False, verbose=False):  
    s = gen_system_state(sim_time, rads_count, arr_rates, urg_times, constant_rads, cutoff, verbose)
    s.run_simulation()    
    return s


def plot_queue_lengths(s):
    fig, ax = plt.subplots()
    for i in range(len(s.queue_lengths[0])):
        plt.plot(s.time_steps, [item[i] for item in s.queue_lengths])

        
def wait_time_plot(img_table):
    fig, ax = plt.subplots()
    x = img_table['create_time'].values
    y = img_table['img_wait_time'].values
    colormap = {
        1: "red",
        2: "orange",
        3: "green"
               }
    colors_list = img_table['urgency'].map(colormap)
    wt_plot = plt.figure()
    plt.scatter(x, y, color=colors_list, alpha=.6)
    plt.xlabel("Time Job Begins")
    plt.ylabel("Time Until Job Seen")   
    plt.title("Time Before Job is Seen")
    red_patch = mpatches.Patch(color='red', alpha=.8, label='Urgency 1')
    orange_patch = mpatches.Patch(color='orange', alpha=.8, label='Urgency 2')
    green_patch = mpatches.Patch(color='green', alpha=.8, label='Urgency 3')
    plt.legend(handles=[red_patch, orange_patch, green_patch])
    plt.show()
    wait_time_plot_url = py.plot_mpl(wt_plot, filename="wait_time_plot")
    return wait_time_plot_url
    

def total_time_plot(img_table):
    x=img_table['create_time'].values
    y=img_table['total_time'].values
    plt.scatter(x, y)
    
        
def plt_mean_queue_length(s_list):
    fig, ax = plt.subplots()
    for s in s_list:
        plt.plot(s.time_steps, pd.DataFrame(s.queue_lengths).sum(axis=1), label=f"{len(s.rads)}")
    plt.xlabel("time")
    plt.ylabel("Mean Queue Length")
    plt.legend()
    plt.show()
    

def rad_idle_plot(rad):
    idle_times = rad.idle_times
    busy_times = rad.busy_times
    left_var = 0
    for i in range(len(idle_times)):
        plt.barh(rad.rad_id, idle_times[i], left=left_var, color="red")
        left_var = left_var + idle_times[i]
        if i < len(busy_times):
                plt.barh(rad.rad_id, busy_times[i], left=left_var, color="orange")
                left_var = left_var + busy_times[i]

    plt.legend(["idle", "busy"], title="Idle Times", loc="upper right")
    plt.show()
    
    
def idle_plots(rads):
    plot_list = {}
    total_idle = 0
    total_busy = 0
    avg_busy_times = {}
    #idl_plt = plt.figure()
    plt.figure(figsize = (13,10))
    for rad in rads:
        idle_times = rad.idle_times
        busy_times = rad.busy_times
        left_var = 0
        for i in range(len(idle_times)):
            plt.barh(rad.rad_id, idle_times[i], left=left_var, color="red")
            left_var = left_var + idle_times[i]
            if i < len(busy_times):
                plt.barh(rad.rad_id, busy_times[i], left=left_var, color="orange")
                left_var = left_var + busy_times[i]
        per_busy = np.sum(busy_times)/(np.sum(busy_times) + np.sum(idle_times))
        avg_busy_times[rad.rad_id] = per_busy
        total_idle += np.sum(idle_times)
        total_busy += np.sum(busy_times)
        #print(f"Radiologist {rad.rad_id} was busy {round(per_busy, 3)} of the time.")
        plt.legend(["idle", "busy"], title="Idle Times", loc="upper right")
    plt.xlabel("time")
    plt.ylabel("Radiologist ID")
    plt.savefig('figures/idle_plot.jpg', dpi=100)
    
    total_per_busy = total_busy/(total_busy + total_idle)
    # Busy percent plots
    plt.figure(figsize = (10,10))
    plt.bar(avg_busy_times.keys(), avg_busy_times.values())
    plt.xlabel("Radiologist ID")
    plt.ylabel("Ratio of Time Busy")
    plt.savefig('figures/busy_plot.jpg', dpi=100)
    #plt.show()
    #print(f"Radiologists were busy {round(total_per_busy,3)} of the time.")
    return total_per_busy#, plotly.tools.mpl_to_plotly(idl_plt)

    
    
def total_time_hist(curr_sim):
    #all urgencies
    """""
    fig, ax = plt.subplots()
    plt.hist(curr_sim.img_table.total_time)
    plt.xlabel("time")
    plt.ylabel("Number of Images")
    plt.title("Time to be processed from creation (All Medical Images)")
    plt.show()
    """""
    
    #urgencies seperated
    fig, ax = plt.subplots()
    plt.hist(curr_sim.img_table[curr_sim.img_table.urgency==3].total_time, label=f"urgency 3: {len(curr_sim.img_table[curr_sim.img_table.urgency==3])} images", color="red", alpha=0.6)
    plt.hist(curr_sim.img_table[curr_sim.img_table.urgency==2].total_time, label=f"urgency 2: {len(curr_sim.img_table[curr_sim.img_table.urgency==2])} images", color="yellow", alpha=0.5)
    plt.hist(curr_sim.img_table[curr_sim.img_table.urgency==1].total_time, label=f"urgency 1: {len(curr_sim.img_table[curr_sim.img_table.urgency==1])} images", color="green", alpha=0.5)
    plt.xlabel("time")
    plt.ylabel("Number of Images")
    plt.title("Time to be processed from creation (All Medical Images)")
    plt.legend()
    plt.show()
    
    t1 = np.mean(curr_sim.img_table[curr_sim.img_table.urgency==1].total_time)
    t2 = np.mean(curr_sim.img_table[curr_sim.img_table.urgency==2].total_time)
    t3 = np.mean(curr_sim.img_table[curr_sim.img_table.urgency==3].total_time)
    print(f"The average total time for urgency 1, 2, and 3 medical images are:")
    print(f"Urgency 1: {t1}")
    print(f"Urgency 2: {t2}")
    print(f"Urgency 3: {t3}")
    

def wait_time_hist(curr_sim):
    #all urgencies
    """""
    fig, ax = plt.subplots()
    plt.hist(curr_sim.img_table.img_wait_time)
    plt.xlabel("time (mins)")
    plt.ylabel("Number of Images")
    plt.title("Time to be processed from creation (All Medical Images)")
    plt.show()
    """""
    
    #urgencies seperated
    fig, ax = plt.subplots()
    plt.hist(curr_sim.img_table[curr_sim.img_table.urgency==3].img_wait_time, label=f"urgency 3: {len(curr_sim.img_table[curr_sim.img_table.urgency==3])} images", color="red", alpha=0.6)
    plt.hist(curr_sim.img_table[curr_sim.img_table.urgency==2].img_wait_time, label=f"urgency 2: {len(curr_sim.img_table[curr_sim.img_table.urgency==2])} images", color="yellow", alpha=0.5)
    plt.hist(curr_sim.img_table[curr_sim.img_table.urgency==1].img_wait_time, label=f"urgency 1: {len(curr_sim.img_table[curr_sim.img_table.urgency==1])} images", color="green", alpha=0.5)
    plt.xlabel("time (mins)")
    plt.ylabel("Number of Images")
    plt.title("Time to be processed from creation (All Medical Images)")
    plt.legend()
    plt.show()
    
    t1 = np.mean(curr_sim.img_table[curr_sim.img_table.urgency==1].img_wait_time)
    t2 = np.mean(curr_sim.img_table[curr_sim.img_table.urgency==2].img_wait_time)
    t3 = np.mean(curr_sim.img_table[curr_sim.img_table.urgency==3].img_wait_time)
    print(f"The average wait time for urgency 1, 2, and 3 medical images are:")
    print(f"Urgency 1: {t1}")
    print(f"Urgency 2: {t2}")
    print(f"Urgency 3: {t3}")
    
def completion_plot(sims_dict):
    arr_rates = []
    sims_compl_rates = []
    for arr_val, sim in sims_dict.items():
        compl_num = len(sim.img_table)
        un_fin_num = len(sim.unfin_img_table)
        perc_compl = compl_num/(compl_num + un_fin_num)
        sims_compl_rates.append(perc_compl)
        arr_rates.append(arr_val)
        sim_duration = sim.sim_duration
        #print(f"Arr every {arr_val} had {perc_compl} completion.")
    # Busy percent plots
    fig, ax = plt.subplots()
    plt.scatter(arr_rates, sims_compl_rates)
    plt.xlabel("Average time between image creation (min)")
    plt.ylabel(f"Percent of Images completed in {2*sim_duration} minutes")
    plt.show()
        
             






