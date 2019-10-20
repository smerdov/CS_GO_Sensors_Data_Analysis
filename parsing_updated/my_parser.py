import sys
from demodump import DemoDump

sys.path.append('/Users/asm/Projects/CybersportsmenDataAnalysis/parsing_updated/demoinfo-csgo-python/src/demoinfocsgo')

demo = DemoDump() #Create demodump instance
demo.open("pov_2_lp.dem") #open a demo file

#register on events
demo.register_on_netmsg(net_SetConVar, on_list_received) #net_SetConVar = net message id, 2nd param is callback
demo.register_on_gameevent("player_connect", player_connected) #player_connect = game event name (see data/game_events.txt), 2nd param is callback

demo.dump() #start analyzing the demo




fs.readFile("/Users/asm/Projects/CybersportsmenDataAnalysis/parsing_updated/pov_1_lp.dem", (err, buffer) => {
  const demoFile = new demofile.DemoFile();

  demoFile.stringTables.on("update", e => {
    if (e.table.name === "userinfo" && e.userData != null) {
      console.log("\nPlayer info updated:");
      console.log(e.entryIndex, e.userData);
    }
  });

  demoFile.parse(buffer);
});

fs.readFile("/Users/asm/Projects/CybersportsmenDataAnalysis/parsing_updated/pov_3_lp.dem", (err, buffer) => {
  const demoFile = new demofile.DemoFile();

  demoFile.gameEvents.on("player_death", e => {
    const victim = demoFile.entities.getByUserId(e.userid);
    const victimName = victim ? victim.name : "unnamed";

    // Attacker may have disconnected so be aware.
    // e.g. attacker could have thrown a grenade, disconnected, then that grenade
    // killed another player.
    const attacker = demoFile.entities.getByUserId(e.attacker);
    const attackerName = attacker ? attacker.name : "unnamed";

    const headshotText = e.headshot ? " HS" : "";

    console.log(`${attackerName} [${e.weapon}${headshotText}] ${victimName}`);
  });

  demoFile.parse(buffer);
});

fs.readFile("/Users/asm/Projects/CybersportsmenDataAnalysis/parsing_updated/pov_3_lp.dem", (err, buffer) => {
  const demoFile = new demofile.DemoFile();

  demoFile.gameEvents.on("round_end", e => {
    console.log(
      "*** Round ended '%s' (reason: %s, time: %d seconds)",
      demoFile.gameRules.phase,
      e.reason,
      demoFile.currentTime
    );

    // We can't print the team scores here as they haven't been updated yet.
    // See round_officially_ended below.
  });

  demoFile.gameEvents.on("round_officially_ended", e => {
    const teams = demoFile.teams;

    const terrorists = teams[2];
    const cts = teams[3];

    console.log(
      "\tTerrorists: %s score %d\n\tCTs: %s score %d",
      terrorists.clanName,
      terrorists.score,
      cts.clanName,
      cts.score
    );
  });

  demoFile.parse(buffer);
});