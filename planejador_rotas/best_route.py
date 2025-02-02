import googlemaps
import math
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

class BestRouteTSP:
    """
    Find the shortest route that visits all addresses exactly once and returns
    to the starting point.

    Args:
    ------
    API_KEY (str) : API KEY from Google Maps
    starting_point (str): address from the starting point
    addresses (list): delivery address list

    Method:
    -------
    to_dataframe(): Return a dataframe with the delivery info

    Example:
    ----
    >>> df = BestRoute(API_KEY, company_address, delivery_addresses).to_dataframe()
    """

    def __init__(self, API_KEY:str, starting_point:str, addresses:list):
        self.gmaps = googlemaps.Client(key=API_KEY) # Acess the CLient
        self.starting_point = starting_point
        self.address = addresses
        self.route_delivery = self.__delivery_address

    @classmethod
    def __create_tsp_model(cls, distance_matrix):
        """
        Find the shortest route that visits all addresses exactly once and
        returns to the starting point.

        Args:
            distance_matrix (lsit[list[i]]): Matrix with the delivery route distances

        Returns:
            list: Returns the best route
        """
    
        # Creates the routing model
            # 'RountingIndexManager': Creates an index manager for the nodes (addresses)
                # 'len(distance_matrix)': number of addresses (nós)
                # 1: number of vehicles  (in the case of TSP, it is always 1)
                # 0: starting point index (the first address)
            
            # RountingModel: creates the routing model
    
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Defines the cost function (distance)
            # 'distance_callback': function that calculates the distance between two nodes
                # 'from_index' and 'to_index': node indices in the routing model
                # 'manager.IndexToNode': converts model indices into distance matrix indices
                # Returns the distance between nodes 'from_node' and 'to_node' from the distance array
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        # RegisterTransitCallback: Registers the cost function in the routing model
        # SetArcCostEvaluatorOfAllVehicles: defines the cost function as the metric to be minimized
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Configure the search
            # DefaultRoutingSearchParameters: Creates default search parameters
            # first_solution_strategy: Define the strategy for finding an initial solution
                # PATH_CHEAPEST_ARC: Choose the lowest cost edge at each step, building a greedy route
                    # Other more accurate options are GLOBAL_CHEAPEST_ARC ou LOCAL_CHEAPEST_INSERTION.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the Problem
            # SolveWithParameters: Resolves the routing problem using the configured parameters
            # The solution found is stored in the solution variable
        solution = routing.SolveWithParameters(search_parameters)

        # Extract the route
            # routing.Start(0): Starts the route at the starting point (index 0)
            # routing.NextVar(index): Get the next node on the route
            # solution.Value: Returns the value of the decision variable (next node)
            # The loop traverses the route until it reaches the end, adding each node to the route list.
            # The last node is added manually to close the cycle
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

        return route
    
    @property
    def __delivery_address(self):
        """Create the delivery route

        Args:
        -----
            starting (str): Starting Point
            addresses (list): delivery address list

        Returns:
        -----
            list: Returns a list with the starting point added at position 0
        """
        address_list = self.address.copy()
        address_list.insert(0, self.starting_point)
        return address_list
        
    def __calculate_distance_duration(self, origin:str, destination:str):
        """Function to calculate the distance between two addresses

        Args:
            origin (str): Starting Point Address
            destination (str): Destination Address

        Returns:
            int, int: return the distance and duration
        
        Example:
            distance_duration = calculate_distance_duration(origin, destination)
            distance = distance_duration[0]
            duration = distance_duration[1]
        """
        
        # API Call from Google Maps
            # gmaps.distance_matrix: Googlemaps library method that calculates the
            # distance and travel time between two addresses
                # origin: Starting Address
                # destination: Destination Address
                # mode="driving": Mode of transport (can be "driving", "walking",
                # "bicycling" or "transit")
        try:
            info = self.gmaps.distance_matrix(
                origin,
                destination,
                mode="driving"
            )
            if info["status"] == "OK":
                distance = info['rows'][0]['elements'][0]['distance']['value']
                duration = info["rows"][0]["elements"][0]["duration"]["value"]
                return distance, duration
            else:
                raise ValueError(
                    f"Error to calculate distance between {origin} and {destination}")
        except Exception as e:
            raise ValueError(f"Error in API request: {e}")

    def __create_distance_matrix(self):
        """Function to create a matrix of distances between all addresses

        Args:
            addresses (str): Addresses from Delivery Route

        Returns:
            list[list[int]]: Return a matrix of distances between all addresses
        """
        # Matrix Initialization
            # n: number of addresses in the list
            # matrix: an nXn matrix initializes with zeros
                # The matrix is ​​symmetric, since the distance from A to B is the same as that from B to A
                # The main diagonal is zero, since the distance from an address to itself is zero
        n = len(self.route_delivery)
        matrix = [[0] * n for _ in range(n)]
        
        # Completing the Matrix
            # Outer Loop (i): loop through each address in the list
            # Inner loop (j): loops through the remaining addresses (after address i)
                # This avoids redundant calculations since the matrix is ​​symmetric.
            # 'calculate_distance_duration': function that calculates distance and duration from Google Maps API
            # Symmetric assignment
                # matrix[i][j] = distance: stores the distance from address i to address j
                # matrix[j][i] = distance: stores the distance from address j to address i
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.__calculate_distance_duration(
                    self.route_delivery[i], self.route_delivery[j])[0]
                matrix[i][j] = distance
                matrix[j][i] = distance

        return matrix
    
    def __generate_route(self):
        """Generate a dataframe with the best route

        Args:
            addresses (list(str)): Address from Delivery Route

        Returns:
            dataframe: Returns a dataframe with address, distance and duration of the best route
        """
        n_delivery = [] # Stores the number of delivery
        address = [] # Stores the address
        distance = [] # Stores the distance
        duration = [] # Stores the time travel

        # Delivery Route
        route_addresses = self.route_delivery
        
        # Create the distance matrix
        distance_matrix = self.__create_distance_matrix()

        # Execute the nearest neighbor algorithm to find the best route
        route = self.__create_tsp_model(distance_matrix)

        # Calculates the distance and travel time and stores them in the respective list
        for i in range(1, len(route_addresses)+1):
            dist_duration = self.__calculate_distance_duration(
                route_addresses[route[i-1]], route_addresses[route[i]])
            n_delivery.append(i)
            address.append(route_addresses[route[i]])
            distance.append(dist_duration[0])
            duration.append(math.ceil(dist_duration[1]/60))
        
        return n_delivery, address, distance, duration
    
    def to_dataframe(self):
        info = self.__generate_route()
        n_delivery = info[0]
        address = info[1]
        distance = info[2]
        duration = info[3]
        
        # Create a dataframe
        df = pd.DataFrame(
            zip(n_delivery, address, distance, duration),
            columns=["Delivery", "Address", "Distance (m)", "Duration (min)"],
            )
        return df