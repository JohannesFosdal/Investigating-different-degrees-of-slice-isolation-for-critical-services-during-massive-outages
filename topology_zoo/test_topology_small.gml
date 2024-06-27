graph [
multigraph 1
node [
    id 0
    label "UiO"
    Country "Norway"
    Longitude 10.74609
    Internal 1
    Latitude 59.91273
    type "Large Circle as part of major POP"
  ]
  
  node [
    id 1
    label "HiG Gjovik"
    Country "Norway"
    Longitude 10.69155
    Internal 1
    Latitude 60.79574
    type "Small Circle"
  ]

  node [
    id 2
    label "HiBU Kongsberg"
    Country "Norway"
    Longitude 9.65017
    Internal 1
    Latitude 59.66858
    type "Small Circle"
  ]
  
  node [
    id 3
    label "HiBu Honefoss"
    Country "Norway"
    Longitude 10.25647
    Internal 1
    Latitude 60.16804
    type "Small Circle"
  ]
  edge [
    source 0
    target 1
    LinkSpeed "10"
    LinkNote " it/s"
    LinkLabel "10 Gbit/s"
    LinkSpeedUnits "G"
    LinkSpeedRaw 10000000000.0
  ]
  edge [
    source 1
    target 2
    LinkSpeed "10"
    LinkNote " it/s"
    LinkLabel "10 Gbit/s"
    LinkSpeedUnits "G"
    LinkSpeedRaw 10000000000.0
  ]
  edge [
    source 2
    target 3
    LinkSpeed "1"
    LinkNote " it/s"
    LinkLabel "1 Gbit/s"
    LinkSpeedUnits "G"
    LinkSpeedRaw 1000000000.0
  ]
  edge [
    source 3
    target 0
    LinkSpeed "10"
    LinkNote " it/s"
    LinkLabel "10 Gbit/s"
    LinkSpeedUnits "G"
    LinkSpeedRaw 10000000000.0
  ]
]