using System;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels.Ipc;

namespace Linkam
{
    namespace IpcExtension
    {
        /*!
         *  How to decode the data within the LinkData object
         *  -------------------------------------------------
         *  
         *  The ILinkData interface member 'strStageGroup' provides the stage group for the connected stage.
         *  
         * "STDSTAGE"       = Standard Hot Stage
         * "PESTAGE"        = Peltier Stage
         * "GSSTAGE"        = Obsolete.
         * "DSCSTAGE"       = DSC Version 1 Stage
         * "VACSTAGE"       = Vacuum Stage
         * "PSTAGE"         = Pressure Stage
         * "MDSSTAGE"       = Motor Driven Stage (X/Y driven standard stage)
         * "TSTSTAGE"       = Tensile Test Stage (Original force stage)
         * "CSSSTAGE"       = CSS Stage
         * "TCSTAGE"        = Thermocouple Vacuum Stage (High temperature stage)
         * "WARMSTAGE"      = Obsolete.
         * "CORSTAGE"       = CMS196 series Stage
         * "ITOSTAGE"       = Obsolete.
         * "TCVACSTAGE"     = Thermocouple Vacuum Stage (High temperature stage)
         * "TSTV2STAGE"     = Modular Force Stage (supplying the older stage group name for backward compatibility)
         * "DSCV2STAGE"     = DSC version 2 Stage
         * "FDVSTAGE"       = Freeze Drying Stage
         * "NONE"           = No Stage.
         * 
         *  The ILinkData interface member 'strNewDataTitle' provides a list of strings for the name of
         *  the data, the order these are found maps to the order of the data and the data uint arrays.
         *  
         *  The ILinkData interface member 'fNewData' provides a list of float values, each representing
         *  a specific data title mapped by index.
         *  
         *  The ILinkData interface member 'strNewDataUnit' provides a list of strings for the name of
         *  the data unit of measure, each representing a specific data title mapped by index.
         *  
         *  The ILinkData interface member 'u32Status' provides information about how your stage is
         *  configured; see eDATASTATUS for more details.
         */

        /*!
         *  \enum   eDATASTATUS
         *  
         *  This enumerator type provides the various status bits which can be found within the ILinkData
         *  interface member 'u32Status'. You can use these to identify if specific features are enabled
         *  or present in your stage hardware configuration.
         */
        public enum eDATASTATUS
        {
            bUnitsChanged = 1,
            bVacPressureGaugePresent = 2,
            bRhPresent = 4,
            bShowImages = 8,
            bShowIntensity = 16,
            bShowTriggers = 32,
            bShowTasc = 64,
            bShowDSCCal = 128,
            bShowWSC96StageB = 256,
            bShowWSC96Probe = 512,
            bVacPressureGauge2Present = 1024,
            bMFS2NBeam = 2048,
            bMFS20NBeam = 4096,
            bMFS200NBeam = 8192,
            bMFS600NBeam = 16384,
        }

        /*!
         *  \class  IPCClient
         *  \brief  A compatible IPC client class for interfacing to LINK/NEXUS.  
         *  
         */
        public class IPCClient : IDisposable
        {
            #region Members

            private IpcChannel      mChannel    = null;
            private const string    ObjectUrl   = "ipc://LINK/LINKData";

            #endregion Members

            #region Methods

            #region IDisposal

            public void Dispose()
            {
                Stop();
            }

            #endregion IDisposal

            #region Public

            public void Start()
            {
                if (mChannel == null)
                {
                    mChannel = new IpcChannel();

                    // Register the channel.
                    System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(mChannel, false);

                    // Create message link
                    string objectUri;
                    System.Runtime.Remoting.Messaging.IMessageSink messageSink = mChannel.CreateMessageSink(ObjectUrl, null, out objectUri);
                    Console.WriteLine("The URI of the message sink is {0}.", objectUri);
                    if (messageSink != null)
                    {
                        Console.WriteLine(messageSink.GetType().ToString());
                    }
                }
            }

            public void Stop()
            {
                if (mChannel != null)
                {
                    try
                    {
                        System.Runtime.Remoting.Channels.ChannelServices.UnregisterChannel(mChannel);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                    mChannel = null;
                }
            }

            public string GrabHeader()
            {
                string result = "";
                IpcExtension.LinkData LinkData = new IpcExtension.LinkData();

                result += LinkData.strStageGroup + ", " + LinkData.u32Status + Environment.NewLine;
                foreach (string title in LinkData.strNewDataTitle)
                {
                    if (title.Equals(""))
                        continue;

                    result += title + ", ";
                }
                result += Environment.NewLine;

                return result;
            }

            public string GrabData()
            {
                string                result   = "";
                IpcExtension.LinkData LinkData = new IpcExtension.LinkData();

                foreach (float data in LinkData.fNewData)
                {
                    if (float.IsNaN(data))
                        continue;

                    result += data.ToString("0.00") + ", ";
                }
                result += Environment.NewLine;

                return result;
            }

            #endregion Public

            #endregion Methods

            #region Constructors

            public IPCClient()
            {
                WellKnownClientTypeEntry remoteType = new System.Runtime.Remoting.WellKnownClientTypeEntry(typeof(IpcExtension.LinkData), ObjectUrl);
                RemotingConfiguration.RegisterWellKnownClientType(remoteType);
            }

            ~IPCClient()
            {
                Dispose();
            }

            #endregion Constructors
        }

        /*!
         *  \class      ILinkData
         *  
         *  This provides the interface for Link data over the IPC channel. We cannot change the name of this
         *  interface without forcing customers to recompile their product.
         *  Therefore we'll just keep it for Nexus.
         */
        public interface ILinkData
        {
            string      strStageGroup   { get; }
            UInt32      u32Status       { get; }
            string[]    strNewDataTitle { get; }
            float[]     fNewData        { get; }
            string[]    strNewDataUnit  { get; }
        }

        /*!
         *  \class      LinkData
         *  
         *  This provides the implementation for the interface for Link data over the IPC channel.
         *  We cannot change the name of this class without forcing customers to recompile their
         *  product. Therefore we'll just keep it 
         *  for Nexus.
         *  
         *  \Note       This could cause us some greif as for Nexus the stages may yet evolve with
         *              new data and the data may change format. Necxus is capable of adaptable data
         *              formats for recording, unlike LINK which used a specific configured 20 float
         *              element array. Although some elements were repurposed for specific stages
         *              and stage modes. Nexus has data factories which know how to decode the recorded
         *              data within data packets, so the formats can be different and dynamic! This may
         *              cause problems for the IPC service for some customers, and therefore we may need
         *              a method to convert the Nexus data back to a LINK format. Worth thinking about!
         */
        public class LinkData : MarshalByRefObject, ILinkData
        {
            #region Properties

            public string   strStageGroup   { get; private set; }
            public UInt32   u32Status       { get; private set; }
            public string[] strNewDataTitle { get; private set; }
            public float[]  fNewData        { get; private set; }
            public string[] strNewDataUnit  { get; private set; }

            #endregion Properties

            #region Methods

            public LinkData()
            {
            }
            
            public LinkData(string strStageGroup_, UInt32 u32Status_, string[] strNewDataTitle_, float[] fNewData_, string[] strNewDataUnit_)
            {
                strStageGroup   = strStageGroup_;
                u32Status       = u32Status_;
                strNewDataTitle = strNewDataTitle_;
                fNewData        = fNewData_;
                strNewDataUnit  = strNewDataUnit_;
            }

            public void MemberwiseCopyFrom(LinkData src)
            {
                strStageGroup   = src.strStageGroup;
                u32Status       = src.u32Status;
                strNewDataTitle = src.strNewDataTitle;
                fNewData        = src.fNewData;
                strNewDataUnit  = src.strNewDataUnit;
            }

            #endregion Methods
        }
    }
}


