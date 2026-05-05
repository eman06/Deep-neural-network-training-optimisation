import pstats
import sys

# Load the profile stats
stats = pstats.Stats('profile_stats.prof')

# Print the top 50 functions by cumulative time
print("=" * 80)
print("TOP 50 FUNCTIONS BY CUMULATIVE TIME")
print("=" * 80)
stats.sort_stats('cumulative').print_stats(50)

print("\n" + "=" * 80)
print("TOP 30 FUNCTIONS BY TOTAL TIME (not cumulative)")
print("=" * 80)
stats.sort_stats('time').print_stats(30)

print("\n" + "=" * 80)
print("CALLERS OF TOP TIME-CONSUMING FUNCTIONS")
print("=" * 80)
stats.sort_stats('time')
# Print callers of the top 10 most expensive functions
stats.print_callers(10)

print("\n" + "=" * 80)
print("CALLEES OF TOP TIME-CONSUMING FUNCTIONS")
print("=" * 80)
stats.print_callees(10)
